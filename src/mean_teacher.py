"""Mean Teacher training loop for SegFormer — revised version.

A later iteration of the Mean Teacher loop in
notebooks/03_mean_teacher_semi_supervised.ipynb. Changes vs. the notebook:

* cycles the (smaller) labeled loader so every unlabeled image is seen each epoch
* KL-divergence consistency loss against temperature-sharpened teacher targets
* mixed precision (AMP), channels_last tensors, OneCycle LR schedule
* cosine-decaying consistency weight, checkpoint every 5 epochs

NOTE: this revision was not used for the scored submission (0.6174 Dice) and is
kept as-is for reference. See "Known issues" in the README before running it.
"""
import torch
import torch.nn.functional as F
from copy import deepcopy
from itertools import cycle
import torch.cuda.amp as amp

class MeanTeacher:
    def __init__(self, student_model, ema_decay=0.999):
        self.student = student_model
        self.teacher = deepcopy(student_model)
        self.ema_decay = ema_decay
        
        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad_(False)
            param.detach_()
            
    @torch.no_grad()
    def update_teacher(self):
        """Improved EMA update with buffer handling"""
        student_params = dict(self.student.named_parameters())
        teacher_params = dict(self.teacher.named_parameters())
        
        for name in teacher_params:
            teacher_params[name].data.mul_(self.ema_decay).add_(
                student_params[name].data, alpha=1 - self.ema_decay)

class ConsistencyLoss(torch.nn.Module):
    def __init__(self, temperature=0.1, use_sharpening=True):
        super().__init__()
        self.temperature = temperature
        self.use_sharpening = use_sharpening
        
    def forward(self, student_logits, teacher_logits):
        # Sharpen teacher predictions
        if self.use_sharpening:
            teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
            teacher_probs = teacher_probs ** (1/self.temperature)
            teacher_probs = teacher_probs / teacher_probs.sum(dim=1, keepdim=True)
        else:
            teacher_probs = F.softmax(teacher_logits, dim=1)
            
        student_log_probs = F.log_softmax(student_logits, dim=1)
        return F.kl_div(student_log_probs, teacher_probs.detach(), reduction='batchmean')

def train_semi_supervised(labeled_loader, unlabeled_loader, model, num_epochs, device, 
                         base_criterion, learning_rate=1e-4):
    # Initialize components
    mean_teacher = MeanTeacher(model)
    consistency_loss = ConsistencyLoss(temperature=0.2)
    scaler = amp.GradScaler()
    
    # Optimizer configuration
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=learning_rate,
        total_steps=num_epochs * len(labeled_loader),
        pct_start=0.3
    )
    
    # Mixed precision training loop
    for epoch in range(num_epochs):
        model.train()
        mean_teacher.teacher.train()
        running_loss = 0.0
        
        # Use cycle iterator for smaller dataset
        labeled_iter = cycle(labeled_loader) if len(labeled_loader) < len(unlabeled_loader) else labeled_loader
        
        for batch_idx, ((labeled_imgs, labeled_masks), (unlabeled_imgs, _, _)) in enumerate(zip(labeled_iter, unlabeled_loader)):
            # Move data to device with channels_last memory format
            labeled_imgs = labeled_imgs.to(device, memory_format=torch.channels_last)
            labeled_masks = labeled_masks.to(device)
            unlabeled_imgs = unlabeled_imgs.to(device, memory_format=torch.channels_last)
            
            # Mixed precision context
            with amp.autocast():
                # Supervised loss
                labeled_outputs = model(pixel_values=labeled_imgs).logits
                supervised_loss = base_criterion(labeled_outputs, labeled_masks)
                
                # Teacher predictions with weak augmentation
                with torch.no_grad():
                    teacher_outputs = mean_teacher.teacher(pixel_values=unlabeled_imgs).logits
                
                # Student predictions with strong augmentation
                student_outputs = model(pixel_values=unlabeled_imgs).logits
                cons_loss = consistency_loss(student_outputs, teacher_outputs)
                
                # Adaptive consistency weight
                consistency_weight = 0.5 * (1 + np.cos(np.pi * (epoch / num_epochs)))
                total_loss = supervised_loss + consistency_weight * cons_loss
            
            # Gradient scaling and accumulation
            scaler.scale(total_loss).backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            # Update parameters
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            # EMA update
            mean_teacher.update_teacher()
            
            # Learning rate scheduling
            scheduler.step()
            
            running_loss += total_loss.item()
            
            # Free up memory
            del labeled_outputs, student_outputs, teacher_outputs
        
        # Validation and checkpointing
        avg_loss = running_loss / (batch_idx + 1)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save checkpoint with mixed precision state
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'teacher_state_dict': mean_teacher.teacher.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_loss
            }, f'checkpoint_epoch_{epoch+1}.pth', _use_new_zipfile_serialization=False)