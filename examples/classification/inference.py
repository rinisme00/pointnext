import __init__
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from openpoints.models import build_model_from_cfg
from openpoints.utils import EasyConfig, load_checkpoint
from openpoints.dataset import build_dataloader_from_cfg

def plot_confusion_matrix(cm, classes, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True Class',
           xlabel='Predicted Class')

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_point_cloud_2d(points, title, save_path, labels=None):
    # points: (N, 3)
    fig = plt.figure(figsize=(15, 5))
    
    # Side view (X-Y)
    ax1 = fig.add_subplot(131)
    ax1.scatter(points[:, 0], points[:, 1], s=1, c='blue' if labels is None else labels, cmap='viridis' if labels is not None else None)
    ax1.set_title('Side View (X-Y)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.axis('equal')
    
    # Top view (X-Z)
    ax2 = fig.add_subplot(132)
    ax2.scatter(points[:, 0], points[:, 2], s=1, c='blue' if labels is None else labels, cmap='viridis' if labels is not None else None)
    ax2.set_title('Top View (X-Z)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.axis('equal')
    
    # Side view (Y-Z)
    ax3 = fig.add_subplot(133)
    ax3.scatter(points[:, 1], points[:, 2], s=1, c='blue' if labels is None else labels, cmap='viridis' if labels is not None else None)
    ax3.set_title('Side View (Y-Z)')
    ax3.set_xlabel('Y')
    ax3.set_ylabel('Z')
    ax3.axis('equal')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='PointNeXt Inference and Visualization')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--checkpoint', type=str, help='optional checkpoint to load')
    parser.add_argument('--results_dir', type=str, default='results/pointnext_cls', help='directory to save results')
    args = parser.parse_args()

    # Load config
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)

    # Automatically find best checkpoint if not provided
    if args.checkpoint is None:
        log_base = 'log/fantasticbreaks'
        recent_runs = sorted([os.path.join(log_base, d) for d in os.listdir(log_base) if os.path.isdir(os.path.join(log_base, d))], key=os.path.getmtime, reverse=True)
        if not recent_runs:
            print(f"No runs found in {log_base}")
            return
        latest_run = recent_runs[0]
        ckpt_dir = os.path.join(latest_run, 'checkpoint')
        best_ckpts = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if '_best.pth' in f]
        if not best_ckpts:
             print(f"No best checkpoint found in {ckpt_dir}")
             return
        args.checkpoint = best_ckpts[0]
    
    print(f"Loading checkpoint from: {args.checkpoint}")

    # Build model
    model = build_model_from_cfg(cfg.model).cuda()
    load_checkpoint(model, args.checkpoint)
    model.eval()

    # Build dataloader for test set
    cfg.dataset.common.data_dir = os.path.join(os.getcwd(), cfg.dataset.common.data_dir) # ensure absolute path if relative in config
    test_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                            cfg.dataset,
                                            cfg.dataloader,
                                            datatransforms_cfg=cfg.datatransforms,
                                            split='val',
                                            distributed=False)
    classes = cfg.dataset.common.get('classes', ['complete', 'broken'])

    # Run inference
    all_preds = []
    all_targets = []
    all_points = []
    
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'inference_samples'), exist_ok=True)

    from openpoints.models.layers import furthest_point_sample

    print("Running inference on test set...")
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].cuda()
            
            # Point resampling (consistent with train.py)
            points = data['x']
            npoints = cfg.num_points
            num_curr_pts = points.shape[1]
            if num_curr_pts > npoints:
                point_all = int(npoints * 1.2)
                if point_all > num_curr_pts:
                    point_all = num_curr_pts
                fps_idx = furthest_point_sample(points[:, :, :3].contiguous(), point_all)
                fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
                points = torch.gather(points, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, points.shape[-1]))
            
            data['pos'] = points[:, :, :3].contiguous()
            data['x'] = points[:, :, :cfg.model.encoder_args.in_channels].transpose(1, 2).contiguous()

            # Predict
            logits = model(data)
            preds = logits.argmax(dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(data['y'].cpu().numpy())
            all_points.append(data['pos'].cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_points = np.concatenate(all_points)

    # Compute and save metrics
    accuracy = (all_preds == all_targets).mean()
    macc = metrics_macc(all_targets, all_preds, len(classes))
    
    with open(os.path.join(args.results_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Inference Results Summary\n")
        f.write(f"==========================\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Overall Accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"Mean Class Accuracy: {macc * 100:.2f}%\n")
    
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")

    # Save Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_targets, all_preds)
    plot_confusion_matrix(cm, classes, os.path.join(args.results_dir, 'confusion_matrix.png'))
    print(f"Confusion Matrix saved to {os.path.join(args.results_dir, 'confusion_matrix.png')}")

    # Save Point Cloud Visualizations (subset)
    num_samples = 10
    success_count = 0
    fail_count = 0
    
    for i in range(len(all_preds)):
        if all_preds[i] == all_targets[i] and success_count < 5:
            title = f"Sample {i}: Pred={classes[all_preds[i]]}, GT={classes[all_targets[i]]} (SUCCESS)"
            save_path = os.path.join(args.results_dir, 'inference_samples', f'success_{success_count}.png')
            plot_point_cloud_2d(all_points[i], title, save_path)
            success_count += 1
        elif all_preds[i] != all_targets[i] and fail_count < 5:
            title = f"Sample {i}: Pred={classes[all_preds[i]]}, GT={classes[all_targets[i]]} (FAILURE)"
            save_path = os.path.join(args.results_dir, 'inference_samples', f'failure_{fail_count}.png')
            plot_point_cloud_2d(all_points[i], title, save_path)
            fail_count += 1
        
        if success_count >= 5 and fail_count >= 5:
            break
            
    print(f"Inference samples saved to {os.path.join(args.results_dir, 'inference_samples/')}")
    
    # Copy best checkpoint
    import shutil
    shutil.copyfile(args.checkpoint, os.path.join(args.results_dir, 'best_model.pth'))
    print(f"Best model copied to {os.path.join(args.results_dir, 'best_model.pth')}")

def metrics_macc(targets, preds, num_classes):
    class_acc = []
    for c in range(num_classes):
        mask = (targets == c)
        if mask.sum() > 0:
            class_acc.append((preds[mask] == targets[mask]).mean())
    return np.mean(class_acc)

if __name__ == "__main__":
    main()
