import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils

import argparse
import random
from tqdm import tqdm

import wandb


from models import weights_init, Discriminator, Generator
from operation import copy_G_params, load_params, get_dir
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment
policy = 'color,translation'
import lpips
percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)


#torch.backends.cudnn.benchmark = True
def get_perm(l) :
    perm = torch.randperm(l)
    while torch.all(torch.eq(perm, torch.arange(l))) :
        perm = torch.randperm(l)
    return perm

def jigsaw(data, k = 8) :
    with torch.no_grad() :
        actual_h = data.size()[2]
        actual_w = data.size()[3]
        h = torch.split(data, int(actual_h/k), dim = 2)
        splits = []
        for i in range(k) :
            splits += torch.split(h[i], int(actual_w/k), dim = 3)
        fake_samples = torch.stack(splits, -1)
        for idx in range(fake_samples.size()[0]) :
            perm = get_perm(k*k)
            # fake_samples[idx] = fake_samples[idx,:,:,:,torch.randperm(k*k)]
            fake_samples[idx] = fake_samples[idx,:,:,:,perm]
        fake_samples = torch.split(fake_samples, 1, dim=4)
        merged = []
        for i in range(k) :
            merged += [torch.cat(fake_samples[i*k:(i+1)*k], 2)]
        fake_samples = torch.squeeze(torch.cat(merged, 3), -1)
        return fake_samples

def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]

def train_d(net, data, scaler, label="real"):
    """Train function of discriminator"""
    if label=="real":
        part = random.randint(0, 3)
        with torch.cuda.amp.autocast():
            pred, [rec_all, rec_small, rec_part] = net(data, label, part=part)

            err_pred = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 -  pred).mean() 
            err_rec_all = percept( rec_all, F.interpolate(data, rec_all.shape[2]) ).sum()
            err_rec_small = percept( rec_small, F.interpolate(data, rec_small.shape[2]) ).sum()
            err_rec_part = percept( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()

            err = err_pred + err_rec_all + err_rec_small + err_rec_part
            
        scaler.scale(err).backward()
        return pred.mean().item(), err ,err_pred, err_rec_all, err_rec_small, err_rec_part, rec_all, rec_small, rec_part
    elif label == 'real_fake':
        with torch.cuda.amp.autocast():
            pred = net(data, label)
            err = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean() * 0.2 # scale loss
        scaler.scale(err).backward()
        return pred.mean().item()
    else:
        with torch.cuda.amp.autocast():
            pred = net(data, label)
            err = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        scaler.scale(err).backward()
        return pred.mean().item()
        

def train(args):

    data_root = args.path
    total_iterations = args.iter
    checkpoint = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size
    ndf = 64
    ngf = 64
    nz = 256
    nlr = 0.0002
    nbeta1 = 0.5
    use_cuda = True
    multi_gpu = True
    dataloader_workers = 8
    current_iteration = 0
    save_interval = 100
    saved_model_folder, saved_image_folder = get_dir(args)
    
    gscaler=torch.cuda.amp.GradScaler()

    device = torch.device("cpu")
    if use_cuda:
        device = torch.device("cuda:0")

    transform_list = [
            transforms.Resize((int(im_size),int(im_size))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    trans = transforms.Compose(transform_list)
    
    if 'lmdb' in data_root:
        from operation import MultiResolutionDataset
        dataset = MultiResolutionDataset(data_root, trans, 1024)
    else:
        dataset = ImageFolder(root=data_root, transform=trans)

    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers, pin_memory=True))
    '''
    loader = MultiEpochsDataLoader(dataset, batch_size=batch_size, 
                               shuffle=True, num_workers=dataloader_workers, 
                               pin_memory=True)
    dataloader = CudaDataLoader(loader, 'cuda')
    '''
    
    
    #from model_s import Generator, Discriminator
    netG = Generator(ngf=ngf, nz=nz, im_size=im_size)
    netG.apply(weights_init)

    netD = Discriminator(ndf=ndf, im_size=im_size)
    netD.apply(weights_init)

    netG.to(device)
    netD.to(device)

    avg_param_G = copy_G_params(netG)

    fixed_noise = torch.FloatTensor(8, nz).normal_(0, 1).to(device)
    
    if checkpoint != 'None':
        ckpt = torch.load(checkpoint)
        netG.load_state_dict(ckpt['g'])
        netD.load_state_dict(ckpt['d'])
        avg_param_G = ckpt['g_ema']
        optimizerG.load_state_dict(ckpt['opt_g'])
        optimizerD.load_state_dict(ckpt['opt_d'])
        current_iteration = int(checkpoint.split('_')[-1].split('.')[0])
        del ckpt
        
    if multi_gpu:
        netG = nn.DataParallel(netG.to(device))
        netD = nn.DataParallel(netD.to(device))

    optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    
    for iteration in tqdm(range(current_iteration, total_iterations+1)):
        real_image = next(dataloader)
        real_image = real_image.to(device)
        current_batch_size = real_image.size(0)
        noise = torch.Tensor(current_batch_size, nz).normal_(0, 1).to(device)

        with torch.cuda.amp.autocast():
            fake_images = netG(noise)

        real_image = DiffAugment(real_image, policy=policy)
        fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]
        
        ## 2. train Discriminator
        netD.zero_grad()

        err_dr, err_back, err_pred, err_rec_all, err_rec_small, err_rec_part, rec_img_all, rec_img_small, rec_img_part = train_d(netD, real_image, gscaler ,label="real")
        err_dfake = train_d(netD, [fi.detach() for fi in fake_images], gscaler, label="fake")
        err_drealfake = train_d(netD, jigsaw(real_image, args.jigsaw_k), gscaler, label="real_fake")
        gscaler.step(optimizerD)
        # optimizerD.step()
        
        ## 3. train Generator
        netG.zero_grad()
        with torch.cuda.amp.autocast():
            pred_g = netD(fake_images, "fake")
            err_g = -pred_g.mean()

        gscaler.scale(err_g).backward()
        # err_g.backward()
        # optimizerG.step()
        gscaler.step(optimizerG)
        gscaler.update()

        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        if iteration % 100 == 0:
            print("GAN: loss d: %.5f    loss g: %.5f"%(err_dr, -err_g.item()))

        wandb.log({
            'loss_d': err_dr,
            'loss_d_back': err_back,
            'loss_d_pred': err_pred,
            'loss_d_rec_all': err_rec_all,
            'loss_d_rec_small': err_rec_small,
            'loss_d_rec_part': err_rec_part,
            'loss_d_fake': err_dfake,
            'loss_g': -err_g.item(),
            'loss_d_real_fake': err_drealfake,
        })

        if iteration % (args.interval_save_sample) == 0:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            with torch.no_grad():
                vutils.save_image(netG(fixed_noise)[0].add(1).mul(0.5), saved_image_folder+'/%d.jpg'%iteration, nrow=4)
                vutils.save_image( torch.cat([
                        F.interpolate(real_image, 128), 
                        rec_img_all, rec_img_small,
                        rec_img_part]).add(1).mul(0.5), saved_image_folder+'/rec_%d.jpg'%iteration )
            load_params(netG, backup_para)

            wandb.log({"samples": wandb.Image(saved_image_folder+'/%d.jpg'%iteration),
                    "samples_rec": wandb.Image(saved_image_folder+'/rec_%d.jpg'%iteration)})

        if iteration % (save_interval*50) == 0 or iteration == total_iterations:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            torch.save({'g':netG.state_dict(),'d':netD.state_dict()}, saved_model_folder+'/%d.pth'%iteration)
            load_params(netG, backup_para)
            torch.save({'g':netG.state_dict(),
                        'd':netD.state_dict(),
                        'g_ema': avg_param_G,
                        'opt_g': optimizerG.state_dict(),
                        'opt_d': optimizerD.state_dict()}, saved_model_folder+'/all_%d.pth'%iteration)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')

    parser.add_argument('--path', type=str, default='../lmdbs/art_landscape_1k', help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='test1', help='experiment name')
    parser.add_argument('--iter', type=int, default=50000, help='number of iterations')
    parser.add_argument('--interval_save_sample', type=int, default=500, help='number of iters to save after')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=1024, help='image resolution')
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path if have one')
    parser.add_argument('--amp',action='store_true', help='use mixed precision')
    parser.add_argument('--jigsaw_k', type=int, default=8, help='number of iterations')

    args = parser.parse_args()
    print(args)

    wandb.init(config=args)

    train(args)
