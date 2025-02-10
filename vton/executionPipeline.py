import inspect
import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
from helpers import rescale_noise_cfg, latent_to_image, do_repaint, numpy_to_pil


class Pipeline(object):
    def __init__(
        self,
        model,
        device="cuda",
    ):
        self.vae = model.vae
        self.unet_encoder = model.unet_encoder
        self.unet = model.unet
        self.noise_scheduler = model.noise_scheduler
        self.device = device

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @torch.no_grad()
    def __call__(
        self,
        src_image,
        ref_image,
        mask,
        densepose,
        ref_acceleration=False,
        num_inference_steps=50,
        do_classifier_free_guidance=True,
        guidance_scale=2.5,
        generator=None,
        repaint=False,  # used for virtual try-on
        eta=1.0,
        **kwargs,
    ):
        ## From the data-preprocess part
        src_image = src_image.to(device=self.vae.device, dtype=self.vae.dtype)
        ref_image = ref_image.to(device=self.vae.device, dtype=self.vae.dtype)
        mask = mask.to(device=self.vae.device, dtype=self.vae.dtype)
        densepose = densepose.to(device=self.vae.device, dtype=self.vae.dtype)
        masked_image = src_image * (mask < 0.5)

        # 1. VAE encoding
        with torch.no_grad():
            ## Create Latent Space of masked image & ref image
            ## This process reduce the resolution of the images to fit latent space
            masked_image_latent = self.vae.encode(masked_image).latent_dist.sample()
            ref_image_latent = self.vae.encode(ref_image).latent_dist.sample()
        ## Multiply by an scaling factor
        masked_image_latent = masked_image_latent * self.vae.config.scaling_factor
        ref_image_latent = ref_image_latent * self.vae.config.scaling_factor

        ## Interpolate to fit the latent spaces into the masked and denpose
        mask_latent = F.interpolate(
            mask, size=masked_image_latent.shape[-2:], mode="nearest"
        )
        densepose_latent = F.interpolate(
            densepose, size=masked_image_latent.shape[-2:], mode="nearest"
        )

        # 2. prepare noise
        noise = torch.randn_like(masked_image_latent)
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps
        noise = noise * self.noise_scheduler.init_noise_sigma
        latent = noise

        logging.debug(f"Noiser timesteps: {timesteps}")

        # 3. classifier-free guidance
        if do_classifier_free_guidance:
            masked_image_latent = torch.cat([masked_image_latent] * 2)
            ref_image_latent = torch.cat(
                [torch.zeros_like(ref_image_latent), ref_image_latent]
            )
            mask_latent = torch.cat([mask_latent] * 2)
            densepose_latent = torch.cat([densepose_latent] * 2)

        # 4. Denoising loop (Generative UNET)
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = (
            len(timesteps) - num_inference_steps * self.noise_scheduler.order
        )
        reference_features = None

        if ref_acceleration:
            _, reference_features = self.unet_encoder(
                ref_image_latent,
                timesteps[num_inference_steps // 2],
                encoder_hidden_states=None,
                return_dict=False,
            )
            reference_features = list(reference_features)

        with tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latent if we are doing classifier free guidance
                _latent_model_input = (
                    torch.cat([latent] * 2) if do_classifier_free_guidance else latent
                )
                _latent_model_input = self.noise_scheduler.scale_model_input(
                    _latent_model_input, t
                )

                # prepare the input for the inpainting model
                latent_model_input = torch.cat(
                    [
                        _latent_model_input,
                        mask_latent,
                        masked_image_latent,
                        densepose_latent,
                    ],
                    dim=1,
                )

                ### It will run in each timestep the reference_features
                if not ref_acceleration:
                    _, reference_features = self.unet_encoder(
                        ref_image_latent,
                        t,
                        encoder_hidden_states=None,
                        return_dict=False,
                    )
                    reference_features = list(reference_features)

                # Noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=None,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=None,
                    reference_features=reference_features,
                    return_dict=False,
                )[
                    0
                ]  ## Denoised at nth timestep
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )
                elif guidance_scale > 0.0:
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        masked_image_latent,
                        guidance_rescale=guidance_scale,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latent = self.noise_scheduler.step(
                    noise_pred, t, latent, **extra_step_kwargs, return_dict=False
                )[0]
                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.noise_scheduler.order == 0
                ):
                    progress_bar.update()

        # Decode the final latent
        gen_image = latent_to_image(latent, self.vae)

        ## Post processing
        ## smooth transition between generated and non-generated regions.
        if repaint:
            src_image = (src_image / 2 + 0.5).clamp(0, 1)
            src_image = src_image.cpu().permute(0, 2, 3, 1).float().numpy()
            src_image = numpy_to_pil(src_image)
            mask = mask.cpu().permute(0, 2, 3, 1).float().numpy()
            mask = numpy_to_pil(mask)
            mask = [i.convert("RGB") for i in mask]
            gen_image = [
                do_repaint(_src_image, _mask, _gen_image)
                for _src_image, _mask, _gen_image in zip(src_image, mask, gen_image)
            ]

        return (gen_image,)
