FROM pytorch/pytorch
RUN python3 -m pip install scipy
RUN python3 -m pip install tensorboard
RUN python3 -m pip install opencv-python
RUN python3 -c 'import torch; torch.hub.load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth", progress=True)'
RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y python3-pip
RUN apt-get install -y wget
RUN wget https://raw.githubusercontent.com/mbodenham/distiller/master/requirements.txt && pip install -r requirements.txt
RUN git clone https://github.com/NervanaSystems/distiller.git
RUN cd distiller && pip install -e .
