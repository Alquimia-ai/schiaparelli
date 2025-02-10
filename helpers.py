import cv2
import numpy as np
from numpy import asarray
from PIL import Image, ImageFilter, ImageDraw
import json
import torch
import torch.nn.functional as F
from numpy.linalg import lstsq
import logging
from vton.unet_ref import (
    UNet2DConditionModel as ReferenceUNet,
)
from vton.unet_gen import (
    UNet2DConditionModel as GenerativeUNet,
)
from diffusers import AutoencoderKL, DDPMScheduler  # type: ignore

with open("config.json", "r") as file:
    config = json.load(file)
    PADDING_IMG = config["padding_img"]
    MASK_LABEL_MAP = config["mask_label_map"]
    RESIZE_RES = tuple(config["resize_res"])


def resize_and_center(image, target_width, target_height):
    image = image.convert("RGB")
    img = asarray(image)
    original_height, original_width = img.shape[:2]

    scale = min(target_height / original_height, target_width / original_width)

    height = int(original_height * scale)
    width = int(original_width * scale)

    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

    padded_img = np.ones((target_height, target_width, 3), dtype=np.uint8) * PADDING_IMG

    top = (target_height - height) // 2
    left = (target_width - width) // 2

    padded_img[top : top + height, left : left + width] = resized_img

    return Image.fromarray(padded_img)


def filter_mask(parts: list[str], mask):
    """
    Combine multiple label checks into a single mask.
    The resulting mask is going to have all the parts being passed as parameters
    """
    masks = []

    for part in parts:
        xn = mask == MASK_LABEL_MAP[part]
        masks.append(xn)

    combined_mask = np.logical_or.reduce(masks)
    return combined_mask.astype(np.float32)


def mask_head(parse_array):
    """
    Get the mask corresponding to the head
    """
    return filter_mask(["hat", "hair", "sunglasses", "head", "neck"], parse_array)


def mask_arms(parse_array):
    """
    Get the mask corresponding to the arms
    """
    return filter_mask(["left_arm", "right_arm"], parse_array)


def mask_fixed(parse_array):
    """
    Mask garments to be considered fixed, meaning they are going to be preserved after transformation
    """
    return filter_mask(
        ["left_shoe", "right_shoe", "hat", "sunglasses", "scarf", "bag"], parse_array
    )


def mask_dress(parse_array, parser_mask_fixed, parser_mask_changeable):
    """
    Perform a logic not operation on all the masks relevant to dresses
    """
    parse_mask = filter_mask(["dress", "left_leg", "right_leg"], parse_array)
    parser_mask_changeable += np.logical_and(
        parse_array, np.logical_not(parser_mask_fixed)
    )
    return parse_mask, parser_mask_changeable


def mask_upper_body(parse_array, parser_mask_fixed, parser_mask_changeable):
    """
    Perform a logic not operation on all the masks relevant to upper body
    """
    parse_mask = filter_mask(["upper_clothes"], parse_array)
    parser_mask_changeable += np.logical_and(
        parse_array, np.logical_not(parser_mask_fixed)
    )
    return parse_mask, parser_mask_changeable


def mask_lower_body(parse_array, parser_mask_fixed, parser_mask_changeable):
    """
    Perform a logic not operation on all the masks relevant to lower body
    """
    parse_mask = filter_mask(["pants", "left_leg", "right_leg"], parse_array)
    parser_mask_fixed += filter_mask(
        ["upper_clothes", "left_arm", "right_arm"], parse_array
    )
    parser_mask_changeable += np.logical_and(
        parse_array, np.logical_not(parser_mask_fixed)
    )
    return parse_mask, parser_mask_changeable


def is_articulation_detected(articulation):
    return all(coord > 1 for coord in tuple(articulation))


def refine_mask(mask):
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1
    )
    area = []
    for j in range(len(contours)):
        a_d = cv2.contourArea(contours[j], True)
        area.append(abs(a_d))
    refine_mask = np.zeros_like(mask).astype(np.uint8)
    if len(area) != 0:
        i = area.index(max(area))
        cv2.drawContours(  # type: ignore
            refine_mask,
            contours,
            i,
            color=255,
            thickness=-1,  # type: ignore
        )

    return refine_mask


def hole_fill(img):
    img = np.pad(img[1:-1, 1:-1], pad_width=1, mode="constant", constant_values=0)
    img_copy = img.copy()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)

    cv2.floodFill(img, mask, (0, 0), 255)  # type: ignore
    img_inverse = cv2.bitwise_not(img)
    dst = cv2.bitwise_or(img_copy, img_inverse)
    return dst


class SkipAttnProcessor(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        return hidden_states


def remove_cross_attention(
    unet,
    cross_attn_cls=SkipAttnProcessor,
    self_attn_cls=None,
    cross_attn_dim=None,
    **kwargs,
):
    if cross_attn_dim is None:
        cross_attn_dim = unet.config.cross_attention_dim
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None if name.endswith("attn1.processor") else cross_attn_dim
        )
        hidden_size = None
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])

            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            if self_attn_cls is not None:
                attn_procs[name] = self_attn_cls(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    **kwargs,
                )
            else:
                # retain the original attn processor
                attn_procs[name] = AttnProcessor2_0(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    layer_name=name,
                    **kwargs,
                )
        else:
            attn_procs[name] = cross_attn_cls(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                **kwargs,
            )

    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    return adapter_modules


class AttnProcessor2_0(torch.nn.Module):
    """
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self, hidden_size=None, cross_attention_dim=None, layer_name=None, **kwargs
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        self.layer_name = layer_name
        self.model_type = kwargs.get("model_type", "none")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size,
                channel,
                height,
                width,  # type: ignore
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def latent_to_image(latent, vae):
    latent = 1 / vae.config.scaling_factor * latent
    image = vae.decode(latent).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = numpy_to_pil(image)
    return image


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def do_repaint(person, mask, result):
    _, h = result.size
    kernal_size = h // 100
    if kernal_size % 2 == 0:
        kernal_size += 1
    mask = mask.filter(ImageFilter.GaussianBlur(kernal_size))
    person_np = np.array(person)
    result_np = np.array(result)
    mask_np = np.array(mask) / 255
    repaint_result = person_np * (1 - mask_np) + result_np * mask_np
    repaint_result = Image.fromarray(repaint_result.astype(np.uint8))
    return repaint_result


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = (
        guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    )
    return noise_cfg


def agnostic_mask(
    model_parse,
    keypoint,
    category,
    neck_offset_removal: int = 20,
    dilatation_kernel: tuple = (10, 10),
    dilatation_iterations: int = 5,
    stroke_width_mask: int = 30,
    size=(384, 512),
):
    parse_array = np.array(model_parse)
    pose_data = keypoint["pose_keypoints_2d"]
    pose_data = np.array(pose_data)
    pose_data = pose_data.reshape((-1, 2))

    parse_head = mask_head(parse_array)

    arms = mask_arms(parse_array)

    parser_mask_fixed = mask_fixed(parse_array)

    ## Initially only the background is a changeable mask
    parser_mask_changeable = (parse_array == MASK_LABEL_MAP["background"]).astype(
        np.float32
    )
    parse_mask = np.zeros_like(parse_array)
    ## For each subcategory we append new changeables masks
    if category == "dresses":
        parse_mask, parser_mask_changeable = mask_dress(
            parse_array, parser_mask_fixed, parser_mask_changeable
        )

    elif category == "upper_body":
        parser_mask_fixed += filter_mask(["skirt", "pants"], parse_array)
        parse_mask, parser_mask_changeable = mask_upper_body(
            parse_array, parser_mask_fixed, parser_mask_changeable
        )

    elif category == "lower_body":
        parse_mask = mask_lower_body(
            parse_array, parser_mask_fixed, parser_mask_changeable
        )

    ## To Torch tensors
    parse_head = torch.from_numpy(parse_head)
    parse_mask = torch.from_numpy(parse_mask)
    parser_mask_fixed = torch.from_numpy(parser_mask_fixed)
    parser_mask_changeable = torch.from_numpy(parser_mask_changeable)

    width, height = size

    im_arms = Image.new("L", (width, height))
    arms_draw = ImageDraw.Draw(im_arms)

    # dilation
    parse_mask = parse_mask.cpu().numpy()

    width, height = size
    shoulder_right = tuple(pose_data[2, :2])
    elbow_right = tuple(pose_data[3, :2])
    wrist_right = tuple(pose_data[4, :2])
    shoulder_left = tuple(pose_data[5, :2])
    elbow_left = tuple(pose_data[6, :2])
    wrist_left = tuple(pose_data[7, :2])
    keypoints = []
    if category == "dresses" or category == "upper_body":
        # Define a stroke along the arms that helps separate them from the clothing in the segmentation mask.
        # The stroke helps define where the arms are, ensuring they are not erased along with the clothing.
        if is_articulation_detected(wrist_right) is False:
            if is_articulation_detected(elbow_right) is False:
                keypoints = [wrist_left, elbow_left, shoulder_left, shoulder_right]
            else:
                keypoints = [
                    wrist_left,
                    elbow_left,
                    shoulder_left,
                    shoulder_right,
                    elbow_right,
                ]
        elif is_articulation_detected(wrist_left) is False:
            if is_articulation_detected(elbow_left) is False:
                keypoints = [shoulder_left, shoulder_right, elbow_right, wrist_right]
            else:
                keypoints = [
                    elbow_left,
                    shoulder_left,
                    shoulder_right,
                    elbow_right,
                    wrist_right,
                ]
        else:
            keypoints = [
                wrist_left,
                elbow_left,
                shoulder_left,
                shoulder_right,
                elbow_right,
                wrist_right,
            ]

    arms_draw.line(
        keypoints,
        "white",
        stroke_width_mask,
        "curve",
    )

    # increases the thickness of the line so we ensure if they are around the arms is included in the garment
    if height > 512:
        im_arms = cv2.dilate(  # type: ignore
            np.float32(im_arms),  # type: ignore
            np.ones(dilatation_kernel, np.uint16),
            iterations=dilatation_iterations,
        )
    elif height > 256:
        im_arms = cv2.dilate(  # type: ignore
            np.float32(im_arms),  # type: ignore
            np.ones(tuple([z * 2 for z in dilatation_kernel]), np.uint16),  # type: ignore
            iterations=dilatation_iterations,
        )

    hands = np.logical_and(np.logical_not(im_arms), arms)  # type: ignore
    parse_mask += im_arms
    parser_mask_fixed += hands

    parse_head_2 = torch.clone(parse_head)

    ## Neck Removal
    if category == "dresses" or category == "upper_body":
        points = []
        points.append(shoulder_right)
        points.append(shoulder_left)
        x_coords, y_coords = zip(*points)
        ## Create a straightline between the two shoulders
        ## y=mx
        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords, rcond=None)[0]

        ## Remove every NECK_OFFSET_REMOVAL pixels below the y=mx line between the two shoulders
        for i in range(parse_array.shape[1]):
            y = i * m + c
            parse_head_2[int(y - neck_offset_removal * (height / 512.0)) :, i] = 0

    parser_mask_fixed = np.logical_or(
        parser_mask_fixed, np.array(parse_head_2, dtype=np.uint16)
    )
    parse_mask += np.logical_or(
        parse_mask,
        np.logical_and(
            np.array(parse_head, dtype=np.uint16),
            np.logical_not(np.array(parse_head_2, dtype=np.uint16)),
        ),
    )

    if height > 512:
        parse_mask = cv2.dilate(
            parse_mask,
            np.ones(tuple([z * 2 for z in dilatation_kernel]), np.uint16),
            iterations=dilatation_iterations,
        )
    elif height > 256:
        parse_mask = cv2.dilate(
            parse_mask, np.ones(dilatation_kernel, np.uint16), iterations=5
        )
    else:
        parse_mask = cv2.dilate(
            parse_mask,
            np.ones(tuple([int(z / 2) for z in dilatation_kernel]), np.uint16),
            iterations=dilatation_iterations,
        )

    ## changeable ^ !parse_mask (universe - garment)
    ## This ensures only changeable areas are removed (0)
    parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))
    ## parse_mask v fixed
    ## Checks that only remain areas that are not fixed
    parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
    inpaint_mask = 1 - parse_mask_total
    # White = keep, Black = remove (fill in).
    img = np.where(inpaint_mask, 255, 0)  # 255 RGB Channels
    dst = hole_fill(img.astype(np.uint8))
    dst = refine_mask(dst)
    inpaint_mask = dst / 255 * 1
    mask = Image.fromarray(inpaint_mask.astype(np.uint8) * 255)
    keypoints_img = Image.new("RGB", (width, height), (0, 0, 0))
    keypoints_draw = ImageDraw.Draw(keypoints_img)

    for point in keypoints:
        ## Create an elipse around the keypoints
        ## point[0] = x, point[1] = y
        ## equation of an ellipse = (x-h)^2/a^2 + (y-k)^2/b^2 = 1 where (h,k) is the center of the ellipse
        keypoints_draw.ellipse(
            [(point[0] - 5, point[1] - 5), (point[0] + 5, point[1] + 5)], fill="blue"
        )

    keypoints_draw.line(keypoints, fill="red", width=2)

    return mask, keypoints_img


class SchiaparelliModel(torch.nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "",
        pretrained_model: str = "",
        new_in_channels: int = 12,  # noisy_image: 4 +  mask: 1 + masked_image: 4 + densepose: 3
        height: int = RESIZE_RES[1],
        width: int = RESIZE_RES[0],
        dtype: str = "float16",
    ):
        """
        This code is in charge of the instantiation of all the models regarding VTON process
        """
        super().__init__()
        self.height = height
        self.width = width
        self.build_models(
            pretrained_model_name_or_path,
            pretrained_model,
            new_in_channels,
        )

        if dtype == "float16":
            self.half()

    def replace_conv_out_layer(self, unet_model, new_out_channels):
        original_conv_out = unet_model.conv_out

        if original_conv_out.out_channels == new_out_channels:
            return original_conv_out

        new_conv_out = torch.nn.Conv2d(
            in_channels=original_conv_out.in_channels,
            out_channels=new_out_channels,
            kernel_size=original_conv_out.kernel_size,
            padding=1,
        )
        if new_conv_out is None or new_conv_out.bias is None:
            raise Exception("[SCHIAPARELLI/INSTANTIATE] Conv out layer is None")

        new_conv_out.weight.data.zero_()
        new_conv_out.bias.data[: original_conv_out.out_channels] = (
            original_conv_out.bias.data.clone()
        )
        if original_conv_out.out_channels < new_out_channels:
            new_conv_out.weight.data[: original_conv_out.out_channels] = (
                original_conv_out.weight.data
            )
        else:
            new_conv_out.weight.data[:new_out_channels] = original_conv_out.weight.data[
                :new_out_channels
            ]
        return new_conv_out

    def replace_conv_in_layer(self, unet_model, new_in_channels):
        original_conv_in = unet_model.conv_in

        if original_conv_in.in_channels == new_in_channels:
            return original_conv_in

        new_conv_in = torch.nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=original_conv_in.out_channels,
            kernel_size=original_conv_in.kernel_size,
            padding=1,
        )
        new_conv_in.weight.data.zero_()

        if new_conv_in is None or new_conv_in.bias is None:
            raise Exception("[SCHIAPARELLI/INSTANTIATE] Conv in layer is None")

        new_conv_in.bias.data = original_conv_in.bias.data.clone()
        if original_conv_in.in_channels < new_in_channels:
            new_conv_in.weight.data[:, : original_conv_in.in_channels] = (
                original_conv_in.weight.data
            )
        else:
            new_conv_in.weight.data[:, :new_in_channels] = original_conv_in.weight.data[
                :, :new_in_channels
            ]
        return new_conv_in

    def build_models(
        self,
        pretrained_model_name_or_path: str = "",
        pretrained_model: str = "",
        new_in_channels: int = 12,
    ):
        diffusion_model_type = "sd15"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        logging.debug(f"[INSTANTIATE]:  Using {pretrained_model_name_or_path}")

        # Noise Scheduler (Gaussian Noise)
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="scheduler",
            rescale_betas_zero_snr=False if diffusion_model_type == "sd15" else True,
        )

        logging.debug("[INSTANTIATE]: Noise scheduler set")

        # VAE (Variational AutoEncoder)
        vae_config, vae_kwargs = AutoencoderKL.load_config(
            pretrained_model_name_or_path,
            subfolder="vae",
            return_unused_kwargs=True,
        )
        self.vae = AutoencoderKL.from_config(vae_config, **vae_kwargs).to(  # type: ignore
            device=device,
            dtype=torch.float16,  # type: ignore
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)  # type: ignore

        logging.debug("[INSTANTIATE]: VAE set")
        logging.debug("[INSTANTIATE/VAE]: VAE scale factor {self.vae_scale_factor}")

        ## Reference UNET
        ## Gets key references of the Garment

        unet_config, unet_kwargs = ReferenceUNet.load_config(
            pretrained_model_name_or_path,
            subfolder="unet",
            return_unused_kwargs=True,
        )
        self.unet_encoder = ReferenceUNet.from_config(unet_config, **unet_kwargs).to(
            device=device, dtype=torch.float16
        )
        self.unet_encoder.config.addition_embed_type = None

        logging.debug("[INSTANTIATE]: Reference UNET set")

        ## Generative UNET (Denoiser Diffusion)

        unet_config, unet_kwargs = GenerativeUNet.load_config(
            pretrained_model_name_or_path, subfolder="unet", return_unused_kwargs=True
        )

        self.unet = GenerativeUNet.from_config(unet_config, **unet_kwargs).to(
            device, dtype=torch.float16
        )
        self.unet.config.addition_embed_type = None

        logging.debug("[INSTANTIATE]: Generative UNET set")

        ## We are receiving a channel of 12, stablediffusion standard generative UNET uses 4 channels
        ## We have to adapt the conv_in and conv_out layers

        logging.debug(
            "[INSTANTIATE/UNET] Changing conv_in & conv_out layers to match the 12 channels"
        )

        ## TODO: REFACTOR THIS
        unet_conv_in_channel_changed = self.unet.config.in_channels != new_in_channels

        if unet_conv_in_channel_changed:
            self.unet.conv_in = self.replace_conv_in_layer(
                self.unet, new_in_channels
            ).to(device, dtype=torch.float16)
            self.unet.config.in_channels = new_in_channels
        unet_conv_out_channel_changed = (
            self.unet.config.out_channels != self.vae.config.latent_channels  # type: ignore
        )

        if unet_conv_out_channel_changed:
            self.unet.conv_out = self.replace_conv_out_layer(
                self.unet,
                self.vae.config.latent_channels,  # type: ignore
            )
            self.unet.config.out_channels = self.vae.config.latent_channels  # type: ignore

        unet_encoder_conv_in_channel_changed = (
            self.unet_encoder.config.in_channels != self.vae.config.latent_channels  # type: ignore
        )

        if unet_encoder_conv_in_channel_changed:
            self.unet_encoder.conv_in = self.replace_conv_in_layer(
                self.unet_encoder,
                self.vae.config.latent_channels,  # type: ignore
            )
            self.unet_encoder.config.in_channels = (
                self.vae.config.latent_channels  # type:ignore
            )

        unet_encoder_conv_out_channel_changed = (
            self.unet_encoder.config.out_channels
            != self.vae.config.latent_channels  # type:ignore
        )

        if unet_encoder_conv_out_channel_changed:
            self.unet_encoder.conv_out = self.replace_conv_out_layer(
                self.unet_encoder,
                self.vae.config.latent_channels,  # type:ignore
            )
            self.unet_encoder.config.out_channels = (
                self.vae.config.latent_channels  # type:ignore
            )

        remove_cross_attention(self.unet)
        remove_cross_attention(self.unet_encoder, model_type="unet_encoder")

        logging.debug(
            "[INSTANTIATE/UNET] Conv_in and conv_out layers succesfully  changed!"
        )

        # Load pretrained model
        if pretrained_model and pretrained_model != "":
            # Load the checkpoint, mapping tensors directly to GPU
            checkpoint = torch.load(pretrained_model, map_location=device)

            # Load state dict into your model
            self.load_state_dict(checkpoint)

            self.to(device)

            logging.debug(f"Loaded pretrained model from {pretrained_model}")
