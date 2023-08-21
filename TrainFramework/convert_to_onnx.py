import torch
import torch.onnx
from net.FBODInferenceNet import FBODInferenceBody
import os



def pth_to_onnx(input, checkpoint, onnx_path, input_names=['input'], output_names=['output'], device='cpu'):
    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct,\
              please give a name that ends with \'.onnx\'!')
        return 0
    ### FBODInferenceBody parameters:
    ### input_img_num=5, aggregation_output_channels=16, aggregation_method="multiinput", input_mode="GRG", ### Aggreagation parameters.
    ### backbone_name="cspdarknet53": ### Extract parameters. input_channels equal to aggregation_output_channels.
    model = FBODInferenceBody(input_img_num=5, aggregation_output_channels=16, aggregation_method="multiinput", input_mode="GRG", backbone_name="cspdarknet53")
    model.load_state_dict(torch.load(checkpoint)) #初始化权重
    model.eval()
    model.to(device)
    
    torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names, output_names=output_names) #指定模型的输入，以及onnx的输出路径
    print("Exporting .pth model to onnx model has been successful!")

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    checkpoint = './logs/.pth'
    onnx_path = './test.onnx'
    input = torch.randn(1, 7, 384, 672)
    input = input.cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    pth_to_onnx(input, checkpoint, onnx_path, device=device)