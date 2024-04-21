from Utils.data_utils import load_dataloader_from_generate
from addict import Dict
import torch
from tqdm import tqdm


def test(args, testloader, text_emb):
    """Evaluate the network on the entire test set."""
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            images, labels, text_emb = images.to(args.device), labels.to(args.device), text_emb.to(args.device)
            images, text_emb = images.float(), text_emb.float()
            similarity = images @ (text_emb.t())
            _, predicted_indices = torch.max(similarity, dim=1)
            total += labels.shape[0]
            correct += (predicted_indices == labels).sum().item()

    _accuracy = correct / total
    return _accuracy





if __name__ == "__main__":
    args = Dict()
    args.cfg.model_name = 'ViT-B/32'
    args.cfg.batch_size = 128
    args.device = "cuda"

    dataset_name = 'OrganAMNIST'



    train_loaders, test_loader = load_dataloader_from_generate(args, dataset_name, dataloader_num=1)
    text_emb = torch.load(f'/home/ljz/dataset/{dataset_name}_generated_vitb32/{dataset_name}_vitb32_textemb.pth')

    print(text_emb.size())

    accu = test(args, test_loader, text_emb)
    print(accu)