import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset import CompressedStreamDataset
from models import MVMAE
from utils import setup_logger, get_latest_checkpoint, save_checkpoint, load_checkpoint


def calculate_accuracy(output, target, topk=(1,)):
    """COmputes the accuracy over the top l prediction"""
    with torch.nograd():
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(-1, 1).expand_as(pred))

        res=[]
        for k in topk:
            correct_k = correct[:k].resha[e(-1).float().sum(0, keepdim=True)]
            res.append(correct_k.mul_(100.0/batch_size))
        return res

def main(args):
    logger = setup_logger(args.checkpoint_dir, log_filename="training.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Loading dataset...")
    train_dataset=CompressedStreamDataset(
        data_root = args.train, gop_size=args.gop_size)
    val_dataset=CompressedStreamDataset(
        data_root = args.val, gop_size=args.gop_size)

    logger.info(f"Number of training samples: {len(train_dataset)}")
    logger.info(f"Number of validation samples: {len(val_dataset)}")

    logger.info("Initializing the MV_MAE model...")
    model = MVMAE(
        num_classes=args.num_classes,
        model_zoo_path=args.model_zoo_path,
        fusion_hidden_dim=args.fusion_hidden_dim,
        dropout=args.dropout
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, lr-args.lr, weight_decay = args.weight_decay)

    ### RESUME CHECKPOINT IF EXISTS
    start_epoch=0
    best_acc = 0.0
    latest_ckpt = get_latest_checkpoint(args.checkpoint_dir)
    if latest_ckpt and args.resume:
        logger.info("Found checkpoint : {latest_ckpt}")
        start_epoch, best_acc = load_checkpoint(latest_ckpt, model, optimizer, device=device)

    
    logger.info("=========================================")
    logger.info("          STARTING TRAINING              ")
    logger.info("=========================================")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        train_top1 = 0.0

        for batch_idx, (iframes, mvs, labels) in enumerate(train_loader):
            iframes, mvs, labels = iframes.to(device), mvs.to(device), labels.to(device)

            # forward pass
            logits = model(iframes, mvs)
            loss = criterion(logits, labels)

            # backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update metrics
            acc1, acc5 = calculate_accuracy(logits, labels, topk=(1,5))
            train_loss += loss.item()
            train_top1 += acc1

            if batch_idx % args.log_interval == 0:
                logger.info(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                            f"Loss: {loss.item():.4f} | Top-1 Acc: {acc1:.2f}%")

        # validation phase
        model.eval()
        val_loss = 0.0
        val_top1 = 0.0

        with torch.no_grad():
            for iframes, mvs, labels in val_loader:
                iframes, mvs, labels = iframes.to(device), mvs.to(device), labels.to(device)

                logits = model(iframes, mvs)
                loss = criterion(logits, labels)

                acc1, _ = calculate_accuracy(logits, labels, topk=(1,5))
                val_loss += loss.item()
                val_top1 += acc1
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_top1 = val_top1 / len(val_loader)
        

        logger.info(f"--- Epoch {epoch+1} Summary ---")
        logger.info(f"Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")
        logger.info(f"Train Acc:  {train_top1/len(train_loader):.2f}% | Val Acc:  {avg_val_acc:.2f}%")
        
        # 7. Save Checkpoint
        is_best = avg_val_acc > best_acc
        if is_best:
            best_acc = avg_val_acc
            
        state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
        }
        save_checkpoint(state, is_best, ckpt_dir=args.checkpoint_dir, filename=f"checkpoint_epoch_{epoch+1}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GreenEye MV-MAE Training")
    parser.add_argument('--train_data', type=str, default='./dataset/train', help='Path to training data')
    parser.add_argument('--val_data', type=str, default='./dataset/val', help='Path to validation data')
    parser.add_argument('--model_zoo', type=str, default='./model_zoo', help='Path to pretrained weights')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save logs/weights')
    
    parser.add_argument('--num_classes', type=int, default=155, help='Number of action classes (UAV-Human=155)')
    parser.add_argument('--gop_size', type=int, default=16, help='Size of the motion vector GOP')
    
    parser.add_argument('--epochs', type=int, default=50, help='Total training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay for AdamW')
    
    parser.add_argument('--num_workers', type=int, default=8, help='Dataloader workers')
    parser.add_argument('--log_interval', type=int, default=20, help='Print logs every N batches')
    parser.add_argument('--resume', action='store_true', default=True, help='Auto-resume from latest checkpoint')

    args = parser.parse_args()
    main(args)