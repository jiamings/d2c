import torch
import argparse

parser = argparse.ArgumentParser(description='Converting checkpoint')
parser.add_argument('--AE_ckpt', type=str, default='', required=True,
                        help='path to ckpt for the Auto-Encoder')
parser.add_argument('--DDIM_ckpt', type=str, default='', required=True,
                        help='path to ckpt for the DDIM model')
parser.add_argument('--latent_ckpt', type=str, default='', required=True,
                        help='path to ckpt containing latent stats')
parser.add_argument('--output_path', type=str, default='', required=True,
                        help='path to output ckpt')
args = parser.parse_args()

ae_state_dict = torch.load(args.AE_ckpt)["state_dict"]
ae_state_dict_modified = {}

for key in ae_state_dict:
	ae_state_dict_modified[key.replace("module.", "")] = ae_state_dict[key]

del ae_state_dict
print("ae ckpt done")
ddim_state_dict = torch.load(args.DDIM_ckpt)[-1]
print("ddim ckpt done")

latent_ckpt = torch.load(args.latent_ckpt)
print("latent ckpt done")

final_ckpt = {'autoencoder': ae_state_dict_modified, 'diffusion': ddim_state_dict, 'latent_mod':latent_ckpt}
# final_ckpt = {'latent_mod':latent_ckpt}

torch.save(final_ckpt, args.output_path)