from __future__ import print_function
#%matplotlib inline
import random
from typing import Any, Callable, Optional
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils
import struct
from PIL import Image
#import pyarrow as pa
import os

color_map = [
            (0, 0, 0),			# Empty				Black
            (255, 0, 0),		# Blocks			Red
            (0, 255, 0),		# Background		Green
            (0, 0, 255),		# UpArrow			Blue
            (255, 255, 0),		# LeftArrow			Yellow
            (0, 255, 255),		# RightArrow		Cyan
            (255, 0, 255),		# AntiGravityArrow	Magenta
            (255, 255, 255),	# SlowAntiGravity	White
            (255, 120, 120),	# Coin				Salmon
            (120, 255, 120),	# GrabbableObjects	Mint
            (120, 120, 255),	# Hazards			Baby Blue
            ]

block_map = [
			Image.open('pyimg/empty.png'),
   			Image.open('pyimg/blocks.png'),
			Image.open('pyimg/background.png'),
			Image.open('pyimg/uparrow.png'),
			Image.open('pyimg/leftarrow.png'),
			Image.open('pyimg/rightarrow.png'),
			Image.open('pyimg/antigravityarrow.png'),
			Image.open('pyimg/slowantigravity.png'),
			Image.open('pyimg/coin.png'),
			Image.open('pyimg/grabbableobjects.png'),
			Image.open('pyimg/hazards.png'),
			]

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and 1 > 0) else "cpu")

# def preload_lvls_as_pyarrow(src: str, dst: str):
#     src_files = os.listdir(src)
#     for file in src_files:
#         lvl = lvl_loader_cold(file)
#         tensor = pa.Tensor.from_numpy(lvl.numpy())

#         with pa.OSFile(dst + file, 'w') as file:
#             pa.ipc.write_tensor(tensor, file)

# src_dir = "C:\\Users\\awt24\\Downloads\\ArchivEE_1.0.0\\onehot\\output\\"
# dst_dir = "C:\\Users\\awt24\\Downloads\\ArchivEE_1.0.0\\onehot\\c_pyarrow\\"
# preload_lvls_as_pyarrow(src_dir, dst_dir)

# def lvl_loader(path: str) -> torch.FloatTensor:
#     with pa.memory_map(path, 'r') as file:
#         pyarrow_tensor = pa.ipc.read_tensor(file)
#     return torch.FloatTensor(pyarrow_tensor.to_numpy())

def lvl_loader(path: str) -> torch.FloatTensor:
    #print('loading...', path)
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as file:
        width = struct.unpack("<i", file.read(4))[0]
        height = struct.unpack("<i", file.read(4))[0]
        res = np.zeros((11, height, width))

        blocks = file.read()

        x, y = 0, 0
        for block in blocks:
            res[block, y, x] = 1
            x += 1
            if x == width:
                x = 0
                y += 1

        #print('loaded!~', path)
        lvl = torch.from_numpy(res).float()
        return lvl

def visualize(t: torch.FloatTensor) -> Image.Image:
    c, h, w = t.size()
    img = Image.new("RGB", (w*16, h*16))
    
    for x in range(w):
        for y in range(h):
            max_val = -1000
            typ = -1
            for idx in range(11):
                if t[idx, y, x] > max_val:
                    typ = idx
                    max_val = t[idx, y, x]
            img.paste(block_map[typ], (x*16, y*16))

    return img

"""
level = lvl_loader("C:\\Users\\awt24\\Downloads\\ArchivEE_1.0.0\\onehot\\output\\4 elements-xje.lvl")
img = visualize(level)
img.save("visualized.png")

import os
os.exit(0)
"""

"""
level = lvl_loader("C:\\Users\\awt24\\Downloads\\ArchivEE_1.0.0\\output\\4 elements-xje.lvl")
print(level)

import os
os.exit(0)
"""

class LevelFolder(dset.DatasetFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = lvl_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            loader,
            (".lvl") if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.lvls = self.samples


def main():
	print(torch.cuda.is_available())
	gen_file = ""
	opt_gen_file = ""
	disc_file = ""
	opt_disc_file = ""

	# Set random seed for reproducibility
	manualSeed = 1000
	#manualSeed = random.randint(1, 10000) # use if you want new results
	print("Random Seed: ", manualSeed)
	random.seed(manualSeed)
	torch.manual_seed(manualSeed)

	# Root directory for dataset
	dataroot = "data/celeba"

	# Number of workers for dataloader
	workers = 16

	# Batch size during training
	batch_size = 128

	# Spatial size of training images. All images will be resized to this
	#   size using a transformer.
	image_size = 64

	# Number of channels in the training images. For color images this is 3
	nc = 11

	# Size of z latent vector (i.e. size of generator input)
	nz = 100

	# Size of feature maps in generator
	ngf = 64

	# Size of feature maps in discriminator
	ndf = 64

	# Number of training epochs
	num_epochs = 1000

	# Learning rate for optimizers
	lr = 0.0002

	# Beta1 hyperparam for Adam optimizers
	beta1 = 0.5

	# Number of GPUs available. Use 0 for CPU mode.
	ngpu = 1

	png_dir = "C:\\Users\\awt24\\Downloads\\ArchivEE_1.0.0\\onehot\\"

	# Define any desired transformations
	transformations = transforms.Compose([
		transforms.RandomCrop((image_size, image_size)),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	batch_size = 64

	# We can use an image folder dataset the way we have it setup.
	# Create the dataset
 
	dataset = LevelFolder(root=png_dir,
								transform=transforms.Compose([
									transforms.RandomCrop((image_size, image_size)),
								]))
 
	# Create the dataloader
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
											shuffle=True, num_workers=workers)

	# custom weights initialization called on netG and netD
	def weights_init(m):
		classname = m.__class__.__name__
		if classname.find('Conv') != -1:
			nn.init.normal_(m.weight.data, 0.0, 0.02)
		elif classname.find('BatchNorm') != -1:
			nn.init.normal_(m.weight.data, 1.0, 0.02)
			nn.init.constant_(m.bias.data, 0)
			
	# Generator Code

	class Generator(nn.Module):
		def __init__(self, ngpu):
			super(Generator, self).__init__()
			self.ngpu = ngpu
			self.main = nn.Sequential(
				# input is Z, going into a convolution
				nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
				nn.BatchNorm2d(ngf * 8),
				nn.ReLU(True),
				# state size. (ngf*8) x 4 x 4
				nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
				nn.BatchNorm2d(ngf * 4),
				nn.ReLU(True),
				# state size. (ngf*4) x 8 x 8
				nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
				nn.BatchNorm2d(ngf * 2),
				nn.ReLU(True),
				# state size. (ngf*2) x 16 x 16
				nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
				nn.BatchNorm2d(ngf),
				nn.ReLU(True),
				# state size. (ngf) x 32 x 32
				nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
				nn.Tanh()
				# state size. (nc) x 64 x 64
			)

		def forward(self, input):
			return self.main(input)
	

	class Discriminator(nn.Module):
		def __init__(self, ngpu):
			super(Discriminator, self).__init__()
			self.ngpu = ngpu
			self.main = nn.Sequential(
				# input is (nc) x 64 x 64
				nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
				nn.LeakyReLU(0.2, inplace=True),
				# state size. (ndf) x 32 x 32
				nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
				nn.BatchNorm2d(ndf * 2),
				nn.LeakyReLU(0.2, inplace=True),
				# state size. (ndf*2) x 16 x 16
				nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
				nn.BatchNorm2d(ndf * 4),
				nn.LeakyReLU(0.2, inplace=True),
				# state size. (ndf*4) x 8 x 8
				nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
				nn.BatchNorm2d(ndf * 8),
				nn.LeakyReLU(0.2, inplace=True),
				# state size. (ndf*8) x 4 x 4
				nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
				nn.Sigmoid()
			)

		def forward(self, input):
			return self.main(input)
 	
	# Create the generator
	netG = Generator(ngpu).to(device)

	# Handle multi-gpu if desired
	if (device.type == 'cuda') and (ngpu > 1):
		netG = nn.DataParallel(netG, list(range(ngpu)))

	# Apply the weights_init function to randomly initialize all weights
	#  to mean=0, stdev=0.02.
	netG.apply(weights_init)

	# Print the model
	print(netG)

	# Create the Discriminator
	netD = Discriminator(ngpu).to(device)

	# Handle multi-gpu if desired
	if (device.type == 'cuda') and (ngpu > 1):
		netD = nn.DataParallel(netD, list(range(ngpu)))

	# Apply the weights_init function to randomly initialize all weights
	#  to mean=0, stdev=0.2.
	netD.apply(weights_init)

	# Print the model
	print(netD)
 
	# Setup Adam optimizers for both G and D
	optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
	optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
 
	if gen_file != "":
		optimizerD.load_state_dict(torch.load(opt_disc_file))
		optimizerG.load_state_dict(torch.load(opt_gen_file))
		netG.load_state_dict(torch.load(gen_file))
		netD.load_state_dict(torch.load(disc_file))
		

	# Initialize BCELoss function
	criterion = nn.BCELoss()

	# Create batch of latent vectors that we will use to visualize
	#  the progression of the generator
	fixed_noise = torch.randn(64, nz, 1, 1, device=device)

	# Establish convention for real and fake labels during training
	real_label = 1.
	fake_label = 0.

	# Training Loop

	# Lists to keep track of progress
	img_list = []
	G_losses = []
	D_losses = []
	iters = 0

	print("Starting Training Loop...")
	# For each epoch
	for epoch in range(num_epochs):
		# For each batch in the dataloader
		for i, data in enumerate(dataloader, 0):

			############################
			# (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
			###########################
			## Train with all-real batch
			netD.zero_grad()
			# Format batch
			real_cpu = data[0].to(device)
			b_size = real_cpu.size(0)
			label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
			# Forward pass real batch through D
			output = netD(real_cpu).view(-1)
			# Calculate loss on all-real batch
			errD_real = criterion(output, label)
			# Calculate gradients for D in backward pass
			errD_real.backward()
			D_x = output.mean().item()

			## Train with all-fake batch
			# Generate batch of latent vectors
			noise = torch.randn(b_size, nz, 1, 1, device=device)
			# Generate fake image batch with G
			fake = netG(noise)
			label.fill_(fake_label)
			# Classify all fake batch with D
			output = netD(fake.detach()).view(-1)
			# Calculate D's loss on the all-fake batch
			errD_fake = criterion(output, label)
			# Calculate the gradients for this batch, accumulated (summed) with previous gradients
			errD_fake.backward()
			D_G_z1 = output.mean().item()
			# Compute error of D as sum over the fake and the real batches
			errD = errD_real + errD_fake
			# Update D
			optimizerD.step()

			############################
			# (2) Update G network: maximize log(D(G(z)))
			###########################
			netG.zero_grad()
			label.fill_(real_label)  # fake labels are real for generator cost
			# Since we just updated D, perform another forward pass of all-fake batch through D
			output = netD(fake).view(-1)
			# Calculate G's loss based on this output
			errG = criterion(output, label)
			# Calculate gradients for G
			errG.backward()
			D_G_z2 = output.mean().item()
			# Update G
			optimizerG.step()

			# Output training stats
			if i % 50 == 0:
				print('[%d/%d][%d/%d] (%d)\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
					% (epoch, num_epochs, i, len(dataloader), iters,
						errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

			# Save Losses for plotting later
			G_losses.append(errG.item())
			D_losses.append(errD.item())

			# Check how the generator is doing by saving G's output on fixed_noise
			if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
				with torch.no_grad():
					fake = netG(fixed_noise).detach().cpu()
                
				make_grid(iters, fake)
    
			if (iters % 1000 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
				torch.save(netG.state_dict(), f"netG{iters}.pt")
				torch.save(netD.state_dict(), f"netD{iters}.pt")
				torch.save(optimizerD.state_dict(), f"optimD{iters}.pt")
				torch.save(optimizerG.state_dict(), f"optimG{iters}.pt")

			iters += 1
        

def make_grid(iters: int, fake: torch.Tensor):
    n, c, h, w = fake.size()
    img = Image.new("RGB", ((w+2)*8*16, (h+2)*8*16))
    imX, imY = 0, 0
    for i in range(n):
        sub_img = fake[i]
        im = visualize(sub_img)
        img.paste(im, (imX*(w+1)*16, imY*(h+1)*16))
        imX += 1
        if imX == 8:
            imY += 1
            imX = 0
    img.save(f"visualize_{iters}.png")
    pass

if __name__ == '__main__':
    main()