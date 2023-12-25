from Utils.server_client_utils import SupConLoss, train, test, get_parameters, set_parameters, CategoryEmbedding, train_vae, test_vae, EmbVAE, generate_from_meanvar
import torch
device = 'cuda'
vae_model = EmbVAE(512, 256).to(device)
optimizer_vae = torch.optim.SGD(vae_model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)


def train_by_thresh(index, train_loader_i, test_loader, thresh=5):
    dis, epoch_i = 1000, 0
    while dis > thresh:
        loss_vae = train_vae(train_loader_i, vae_model, optimizer_vae, epoch_i, device)
        sim, dis = test_vae(vae_model, test_loader, device, shuffle=0, noise=0)
        
        print(f'Client[{index}] Epoch[{epoch_i}] {loss_vae = } | {sim = } | {dis = }')
        epoch_i += 1
    return vae_model