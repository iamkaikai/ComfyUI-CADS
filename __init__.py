import torch
COND = 0
UNCOND = 1

class CADS:
    current_step = 0
    last_sigma = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "noise_scale": ("FLOAT", {"min": 0.0, "max": 1.0, "step": 0.01, "default": 0.25}),
                "t1": ("FLOAT", {"min": 0.0, "max": 1.0, "step": 0.01, "default": 0.6}),
                "t2": ("FLOAT", {"min": 0.0, "max": 1.0, "step": 0.01, "default": 0.1}),
            },
            "optional": {
                "rescale_psi": ("FLOAT", {"min": 0.0, "max": 1.0, "step": 0.01, "default": 1.0}),
                "apply_to": (["uncond", "cond", "both"],),
                "key": (["y", "c_crossattn"],),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "do"
    CATEGORY = "utils"

    def do(self, model, noise_scale, t1, t2, rescale_psi=1.0, apply_to="both", key="y"):
        previous_wrapper = model.model_options.get("model_function_wrapper")
        print(f'model: {model}')
        im = model.model.model_sampling
        self.last_sigma = None

        skip = None
        if apply_to == "cond":
            skip = UNCOND
        elif apply_to == "uncond":
            skip = COND

        def cads_gamma(sigma):
            ts = im.timestep(sigma[0])
            t = 1 - round(ts.item() / 999.0, 3)
            if t <= t1:
                return 1.0
            elif t >= t2:
                return 0.0
            return (t2 - t) / (t2 - t1)
            
        def cads_noise(gamma, y):
            if y is None:
                return None

            noise = torch.randn_like(y)
            gamma = torch.tensor(gamma).to(y)
            y_mean, y_std = torch.mean(y), torch.std(y)
            y = gamma.sqrt().item() * y + noise_scale * (1 - gamma).sqrt().item() * noise

            # FIXME: does this work at all like it's supposed to?
            if rescale_psi > 0:
                y_scaled = (y - torch.mean(y)) / torch.std(y) * y_std + y_mean
                if not y_scaled.isnan().any():
                    y = rescale_psi * y_scaled + (1 - rescale_psi) * y
                else:
                    print("Warning, NaNs during rescale")
            return y

        def apply_cads(apply_model, args):
            input_x = args["input"]
            timestep = args["timestep"]
            cond_or_uncond = args["cond_or_uncond"]
            c = args["c"]
           
            if noise_scale > 0.0:
                noise_target = c.get(key, c["c_crossattn"])
                gamma = cads_gamma(timestep)
                for i in range(noise_target.size(dim=0)):
                    if cond_or_uncond[i % len(cond_or_uncond)] == skip:
                        continue
                    noise_target[i] = cads_noise(gamma, noise_target[i])

            if previous_wrapper:
                return previous_wrapper(apply_model, args)

            return apply_model(input_x, timestep, **c)

        m = model.clone()
        m.set_model_unet_function_wrapper(apply_cads)

        return (m,)


NODE_CLASS_MAPPINGS = {"CADS": CADS}
