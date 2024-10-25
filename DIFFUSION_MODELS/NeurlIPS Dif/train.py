import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm


def train_model(model, data, model_params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data = data.to(device)

    optimizer = optim.Adam(model.parameters(), lr=model_params['learning_rate'])
    epochs = 100

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        optimizer.zero_grad()

        noisy_data, _ = model.ts_diff.forward_diffusion_sample(data, t=50)
        predicted_noise = model(noisy_data, torch.tensor([500], dtype=torch.float32).to(device))
        loss = F.mse_loss(predicted_noise, noisy_data)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
        # Save the trained model
    model_path = "ts_diff_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
def evaluate_model(real_fluxes, generated_fluxes):
    if len(real_fluxes.shape) > 1:
        real_fluxes = real_fluxes[0]
    mse = mean_squared_error(real_fluxes.flatten(), generated_fluxes.flatten())
    r2 = r2_score(real_fluxes.flatten(), generated_fluxes.flatten())
    return mse, r2

def print_evaluation(real_fluxes, generated_fluxes):
    mse, r2 = evaluate_model(real_fluxes, generated_fluxes)
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R2 Score): {r2}")
def generate_fluxes(model, wavelength_tensor):
    """
    Generate a new set of fluxes given a set of wavelengths using the trained model.
    Assumes `wavelength_tensor` is of shape [1, num_features, sequence_length].
    """
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        # Simulate noise addition
        noisy_flux, _ = model.ts_diff.forward_diffusion_sample(wavelength_tensor, t=50)  # Example time step
        # Denoise to generate new flux
        generated_flux = model.denoise(noisy_flux, t=500)

    return generated_flux

