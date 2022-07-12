import torch
import torchvision.models as models
from torch.profiler import ProfilerActivity, profile, record_function

if __name__ == "__main__":
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    # # torch.backends.cudnn.enable_tf32 = False
    # torch.backends.cudnn.allow_tf32 = True
    model = models.resnet18()
    model.cuda()
    inputs = torch.randn(5, 3, 224, 224)
    inputs = inputs.cuda()
    model(inputs)
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            for i in range(100):
                model(inputs)

    # print(prof.key_averages().table(sort_by="gpu_time_total", row_limit=10))
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
