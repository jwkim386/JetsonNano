import jetson.inference
import jetson.utils
import argparse

import sys
from depthnet_utils import depthBuffers

# 커맨드 라인 파싱
parser = argparse.ArgumentParser(description="Mono depth estimation on a video/image stream using depthNet DNN.", 
formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.depthNet.Usage() +
jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="fcn-mobilenet", help="pre-trained model to load, see below for options")
parser.add_argument("--visualize", type=str, default="input,depth", help="visualization options (can be 'input' 'depth' 'input,depth'")
parser.add_argument("--depth-size", type=float, default=1.0, help="scales the size of the depth map visualization, as a percentage of the input size (default is 1.0)")
parser.add_argument("--filter-mode", type=str, default="linear", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n 'point' or 'linear' (default: 'linear')")
parser.add_argument("--colormap", type=str, default="viridis-inverted", help="colormap to use for visualization (default is 'viridis-inverted')",
choices=["inferno", "inferno-inverted", "magma", "magma-inverted", "parula", "parula-inverted", 
"plasma", "plasma-inverted", "turbo", "turbo-inverted", "viridis", "viridis-inverted"])

try:
    opt = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# depthnet 모델 불러오기
net = jetson.inference.depthNet(opt.network, sys.argv)

# 버퍼 매니저 생성
buffers = depthBuffers(opt)

# sources & outputs 비디오 생성
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv)

# 종료하기 전까지 프레임을 처리한다.
while True:
    # 다음 이미지 캡처
    img_input = input.Capture()

    # 해당 이미지 크기의 버퍼 메모리를 할당
    buffers.Alloc(img_input.shape, img_input.format)

    # mono depth를 수행하고 visualize도 수행 
    net.Process(img_input, buffers.depth, opt.colormap, opt.filter_mode)

    # 이미지들을 합친다.
    if buffers.use_input:
        jetson.utils.cudaOverlay(img_input, buffers.composite, 0, 0)
    if buffers.use_depth:
        jetson.utils.cudaOverlay(buffers.depth, buffers.composite, img_input.width if buffers.use_input else 0, 0)

    # 결과 이미지를 렝더링
    output.Render(buffers.composite)

    # 타이틀바 업데이트
    output.SetStatus("{:s} | {:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkName(), net.GetNetworkFPS()))

    # 성능 정보를 출력
    jetson.utils.cudaDeviceSynchronize()
    net.PrintProfilerTimes()

    # input/output의 EOS시 exit
    if not input.IsStreaming() or not output.IsStreaming():
        break