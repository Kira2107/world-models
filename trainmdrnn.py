""" Recurrent model training """
import argparse
from functools import partial
from os.path import join, exists
from os import mkdir, makedirs # Combined import
import torch
# Used F.interpolate to replace deprecated F.upsample
import torch.nn.functional as f 
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from utils.misc import save_checkpoint
# NOTE: LSIZE will be overridden below
from utils.misc import ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE 
from utils.learning import EarlyStopping
## WARNING : THIS SHOULD BE REPLACED WITH PYTORCH 0.5
from utils.learning import ReduceLROnPlateau

from data.loaders import RolloutSequenceDataset
from models.vae import VAE
from models.mdrnn import MDRNN, gmm_loss

parser = argparse.ArgumentParser("MDRNN training")
parser.add_argument('--logdir', type=str,
                     help="Where things are logged and models are loaded from.")
parser.add_argument('--noreload', action='store_true',
                     help="Do not reload if specified.")
parser.add_argument('--include_reward', action='store_true',
                     help="Add a reward modelisation term to the loss.")
args = parser.parse_args()

# --- FIXED CONSTANTS ---
# Force to CPU is the recommended fix for your current CUDA error.
device = torch.device('cpu') 
BSIZE = 16
SEQ_LEN = 32
epochs = 30
# FIX: The latent size the VAE checkpoint was saved with (original project value)
VAE_CKPT_LSIZE = 32
# FIX: The latent size required by the data pipeline logic (36864 / (16*32) = 72)
LSIZE_FOR_MDRNN = 72 
# -----------------------

# Loading VAE
vae_file = join(args.logdir, 'vae', 'best.tar')
assert exists(vae_file), "No trained VAE in the logdir..."
# FIX: Use map_location=device to load VAE to CPU regardless of how it was saved
state = torch.load(vae_file, map_location=device) 
print("Loading VAE at epoch {} "
      "with test error {}".format(
          state['epoch'], state['precision']))

# FIX: Instantiate VAE with VAE_CKPT_LSIZE (32) to match checkpoint architecture
vae = VAE(3, VAE_CKPT_LSIZE).to(device) 
vae.load_state_dict(state['state_dict'])

# FIX: Override the global LSIZE variable for the rest of the script 
LSIZE = LSIZE_FOR_MDRNN

# Loading model
rnn_dir = join(args.logdir, 'mdrnn')
rnn_file = join(rnn_dir, 'best.tar')

if not exists(rnn_dir):
    mkdir(rnn_dir)

# MDRNN is instantiated with the corrected LSIZE (72)
mdrnn = MDRNN(LSIZE, ASIZE, RSIZE, 5)
mdrnn.to(device)
# FIX: Lowered learning rate for stability
optimizer = torch.optim.RMSprop(mdrnn.parameters(), lr=1e-4, alpha=.9) 
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
earlystopping = EarlyStopping('min', patience=30)


if exists(rnn_file) and not args.noreload:
    # Use map_location=device to load MDRNN to CPU regardless of how it was saved
    rnn_state = torch.load(rnn_file, map_location=device) 
    print("Reloading MDRNN at epoch {} "
          "with test error {}".format(
              rnn_state["epoch"], rnn_state["precision"]))
    mdrnn.load_state_dict(rnn_state["state_dict"])
    optimizer.load_state_dict(rnn_state["optimizer"])
    scheduler.load_state_dict(rnn_state['scheduler'])
    earlystopping.load_state_dict(rnn_state['earlystopping'])


# Data Loading
transform = transforms.Lambda(
    lambda x: np.transpose(x, (0, 3, 1, 2)) / 255)
train_loader = DataLoader(
    RolloutSequenceDataset('datasets/carracing', SEQ_LEN, transform, buffer_size=30),
    batch_size=BSIZE, num_workers=2, shuffle=True)
test_loader = DataLoader(
    RolloutSequenceDataset('datasets/carracing', SEQ_LEN, transform, train=False, buffer_size=10),
    batch_size=BSIZE, num_workers=2)

# --- CORRECTED TO_LATENT FUNCTION ---
def to_latent(obs, next_obs):
    """ Transform observations to latent space. """
    with torch.no_grad():
        # 1. Flatten the first two dimensions (BSIZE, SEQ_LEN) into one batch dimension (-1)
        flat_obs = obs.contiguous().view(-1, 3, SIZE, SIZE)
        flat_next_obs = next_obs.contiguous().view(-1, 3, SIZE, SIZE)
        
        # 2. Resize and get VAE parameters (vae(x) returns (recon_x, mu, logsigma))
        obs_resized, next_obs_resized = [
            f.interpolate(x, size=RED_SIZE, mode='bilinear', align_corners=True) 
            for x in (flat_obs, flat_next_obs)]

        # 3. VAE forward pass (outputs: _, mu, logsigma)
        # Note: The vae(x) call is wrapped in a list comprehension which correctly unpacks all 3 values.
        (_, obs_mu, obs_logsigma), (_, next_obs_mu, next_obs_logsigma) = [
            vae(x) for x in (obs_resized, next_obs_resized)]

        # 4. Reparameterization Trick: (Output shape is [-1, LSIZE])
        latent_obs = obs_mu + obs_logsigma.exp() * torch.randn_like(obs_mu)
        latent_next_obs = next_obs_mu + next_obs_logsigma.exp() * torch.randn_like(next_obs_mu)
        
        # 5. Final Reshape: Convert back to [BSIZE, SEQ_LEN, LSIZE]
        latent_obs = latent_obs.view(BSIZE, SEQ_LEN, LSIZE)
        latent_next_obs = latent_next_obs.view(BSIZE, SEQ_LEN, LSIZE)
        
    return latent_obs, latent_next_obs
# ------------------------------------


def get_loss(latent_obs, action, reward, terminal,
             latent_next_obs, include_reward: bool):
    """ Compute losses. """
    latent_obs, action,\
        reward, terminal,\
        latent_next_obs = [arr.transpose(1, 0)
                            for arr in [latent_obs, action,
                                        reward, terminal,
                                        latent_next_obs]]
    
    # MDRNN forward pass
    mus, sigmas, logpi, rs, ds = mdrnn(action, latent_obs)
    
    # GMM loss (stabilized in models/mdrnn.py)
    gmm = gmm_loss(latent_next_obs, mus, sigmas, logpi)
    bce = f.binary_cross_entropy_with_logits(ds, terminal)
    
    if include_reward:
        mse = f.mse_loss(rs, reward)
        scale = LSIZE + 2
    else:
        mse = 0
        scale = LSIZE + 1
    loss = (gmm + bce + mse) / scale
    return dict(gmm=gmm, bce=bce, mse=mse, loss=loss)


def data_pass(epoch, train, include_reward): # pylint: disable=too-many-locals
    """ One pass through the data """
    if train:
        mdrnn.train()
        loader = train_loader
    else:
        mdrnn.eval()
        loader = test_loader

    loader.dataset.load_next_buffer()

    cum_loss = 0
    cum_gmm = 0
    cum_bce = 0
    cum_mse = 0

    pbar = tqdm(total=len(loader.dataset), desc="Epoch {}".format(epoch))
    for i, data in enumerate(loader):
        obs, action, reward, terminal, next_obs = [arr.to(device) for arr in data]

        # transform obs
        latent_obs, latent_next_obs = to_latent(obs, next_obs)

        if train:
            losses = get_loss(latent_obs, action, reward,
                              terminal, latent_next_obs, include_reward)

            optimizer.zero_grad()
            losses['loss'].backward()
            
            # --- FIX: GRADIENT CLIPPING FOR STABILITY ---
            # Clip the gradients of all parameters to a maximum norm of 1.0 (or 5.0)
            torch.nn.utils.clip_grad_norm_(mdrnn.parameters(), 5.0) # Set to 5.0 for extra stability
            # ------------------------------------------------
            
            optimizer.step()
        else:
            with torch.no_grad():
                losses = get_loss(latent_obs, action, reward,
                                  terminal, latent_next_obs, include_reward)

        cum_loss += losses['loss'].item()
        cum_gmm += losses['gmm'].item()
        cum_bce += losses['bce'].item()
        cum_mse += losses['mse'].item() if hasattr(losses['mse'], 'item') else \
            losses['mse']

        # NOTE: The division by LSIZE in gmm calculation is intended in the original code
        pbar.set_postfix_str("loss={loss:10.6f} bce={bce:10.6f} "
                              "gmm={gmm:10.6f} mse={mse:10.6f}".format(
                                  loss=cum_loss / (i + 1), bce=cum_bce / (i + 1),
                                  gmm=cum_gmm / LSIZE / (i + 1), mse=cum_mse / (i + 1)))
        pbar.update(BSIZE)
    pbar.close()
    return cum_loss * BSIZE / len(loader.dataset)


train = partial(data_pass, train=True, include_reward=args.include_reward)
test = partial(data_pass, train=False, include_reward=args.include_reward)

cur_best = None
for e in range(epochs):
    train(e)
    test_loss = test(e)
    scheduler.step(test_loss)
    earlystopping.step(test_loss)

    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss
    checkpoint_fname = join(rnn_dir, 'checkpoint.tar')
    save_checkpoint({
        "state_dict": mdrnn.state_dict(),
        "optimizer": optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'earlystopping': earlystopping.state_dict(),
        "precision": test_loss,
        "epoch": e}, is_best, checkpoint_fname,
                        rnn_file)

    if earlystopping.stop:
        print("End of Training because of early stopping at epoch {}".format(e))
        break