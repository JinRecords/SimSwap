import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions


def lcm(a, b): 
    return abs(a * b) / fractions.gcd(a, b) if a and b else 0


# Normalize BOTH images
transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


transformer_Arcface = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


if __name__ == '__main__':
    opt = TestOptions().parse()
    start_epoch, epoch_iter = 1, 0
    torch.nn.Module.dump_patches = True
    model = create_model(opt)
    model.eval()


    with torch.no_grad():
        pic_a = opt.pic_a_path
        img_a = Image.open(pic_a).convert('RGB')
        img_a = transformer_Arcface(img_a)
        img_id = img_a.unsqueeze(0)


        pic_b = opt.pic_b_path
        img_b = Image.open(pic_b).convert('RGB')
        img_b = transformer(img_b)
        img_att = img_b.unsqueeze(0)


        # convert numpy to tensor
        img_id = img_id.cuda()
        img_att = img_att.cuda()


        # create latent id
        img_id_downsample = F.interpolate(img_id, size=(112, 112))
        latend_id = model.netArc(img_id_downsample)
        latend_id = latend_id.detach().to('cpu')
        latend_id = latend_id / np.linalg.norm(latend_id, axis=1, keepdims=True)
        latend_id = latend_id.to('cuda')


        ############## Forward Pass ######################
        img_fake = model(img_id, img_att, latend_id, latend_id, True)


        for i in range(img_id.shape[0]):
            if i == 0:
                row1 = img_id[i]
                row2 = img_att[i]
                row3 = img_fake[i]
            else:
                row1 = torch.cat([row1, img_id[i]], dim=2)
                row2 = torch.cat([row2, img_att[i]], dim=2)
                row3 = torch.cat([row3, img_fake[i]], dim=2)


        full = row3.detach()
        
        # DEBUG: Print output ranges
        print(f"Model output - min: {full.min().item():.4f}, max: {full.max().item():.4f}")
        print(f"Model output - mean: {full.mean().item():.4f}, std: {full.std().item():.4f}")
        full = img_fake.detach().cpu()
        full = torch.clamp(full, 0, 1)
        print(f"After clamp - min: {full.min().item():.4f}, max: {full.max().item():.4f}")

        # PERMUTE from [C, H, W] to [H, W, C]
        full = full.squeeze(0).permute(1, 2, 0)

        # Convert to NumPy
        output = full.numpy()

        # Convert from RGB to BGR for OpenCV
        output = output[..., ::-1]

        # Scale to [0,255]
        output = (output * 255).astype(np.uint8)

        # Save

        # Assume output is [H, W, C] and values are in [0,255], uint8
        # If NOT, run: output = (output * 255).astype(np.uint8)

        # Convert to HSV color space to control saturation
        hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)

        # Reduce saturation by a factor (e.g., 0.6 for 60% original)
        SCALE = 0.8  # Change as desired: 0 = full desaturation (gray), 1 = full original
        hsv[...,1] = (hsv[...,1].astype(np.float32) * SCALE).clip(0,255).astype(np.uint8)

        # Convert back to BGR
        output_desat = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Save the desaturated image
        cv2.imwrite(opt.output_path + 'result.jpg', output_desat)
        print(f"Desaturated image saved to: {opt.output_path}result.jpg")

        from IPython.display import Image, display

    print(f"PATH_A={opt.pic_a_path}")
    print(f"PATH_B={opt.pic_b_path}")
    print(f"RESULT={opt.output_path + 'result.jpg'}")
