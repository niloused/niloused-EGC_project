import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from guided_diffusion.script_util import create_egc_model_and_diffusion, egc_model_and_diffusion_defaults

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading CIFAR-10 datasets
def load_cifar10_data(batch_size):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

# Apply noise and compute q term based on the diffusion model
def apply_noise_and_compute_q(images, timestep, beta_schedule):
    beta = beta_schedule[timestep]
    noise = torch.randn_like(images)
    noised_images = torch.sqrt(1 - beta) * images + torch.sqrt(beta) * noise
    q_loss_term = -(noised_images - torch.sqrt(1 - beta) * images) / beta
    return noised_images, q_loss_term

# Initialize the EGC model with appropriate settings for CIFAR-10
def create_egc_model_for_cifar10():
    defaults = egc_model_and_diffusion_defaults()
    defaults.update({
        'image_size': 32,
        'num_classes': 10,
        'class_cond': True,
        'channel_mult': (1, 2, 2, 2),
        'attention_resolutions': '16,8',
        'num_heads': 4,
        'num_head_channels': 64
    })
    model, diffusion = create_egc_model_and_diffusion(**defaults)
    return model.to(device), diffusion

# Train the EGC model
def train_egc_model(train_loader, model, beta_schedule, epochs, gamma=0.1):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            timestep = torch.randint(0, len(beta_schedule), (1,)).item()
            noised_images, q_loss_term = apply_noise_and_compute_q(images, timestep, beta_schedule)
            pred_recon, pred_labels = model(noised_images)
            recon_loss = (pred_recon - q_loss_term).pow(2).mean()
            class_loss = torch.nn.functional.cross_entropy(pred_labels, labels)
            loss = recon_loss + gamma * class_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'Epoch {epoch+1}, Step {i}, Loss {loss.item()}')

# Main function
def main():
    batch_size = 128
    epochs = 10
    beta_schedule = torch.linspace(0.0001, 0.02, steps=1000)  # Example linear schedule

    train_loader = load_cifar10_data(batch_size)
    model, diffusion = create_egc_model_for_cifar10()
    train_egc_model(train_loader, model, beta_schedule, epochs)

if __name__ == "__main__":
    main()
