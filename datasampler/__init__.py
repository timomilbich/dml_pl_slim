import datasampler.class_random_sampler
import datasampler.random_sampler



def select(name, image_dict, image_list=None, **kwargs):
    if 'class' in name:
        sampler_lib = datasampler.class_random_sampler
    elif 'full' in name:
        raise Exception('Minibatch sampler <{}> not supported, yet!'.format(name))
        # sampler_lib = random_sampler
    else:
        raise Exception('Minibatch sampler <{}> not available!'.format(name))

    sampler = sampler_lib.Sampler(image_dict=image_dict,image_list=image_list, **kwargs)

    return sampler
