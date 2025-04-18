import torch
def evaluate(model, loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model = model.to(device)
    
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, pred_top1 = outputs.topk(1, 1, True, True)
            _, pred_top5 = outputs.topk(5, 1, True, True)

            top1_correct += (pred_top1.squeeze() == labels).sum().item()
            top5_correct += sum([labels[i].item() in pred_top5[i] for i in range(len(labels))])
            total += labels.size(0)
    
    print(f"Top-1 Accuracy: {100 * top1_correct / total:.2f}%")
    print(f"Top-5 Accuracy: {100 * top5_correct / total:.2f}%")
