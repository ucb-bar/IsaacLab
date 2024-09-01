
import numpy as np
import torch

from cc.udp import UDP



ENV_ADDR = ("10.0.0.68", 8010)
POLICY_ADDR = ("0.0.0.0", 8011)


model = torch.load("logs/model.pt", weights_only=False)
model.eval()


udp = UDP(recv_addr=POLICY_ADDR, send_addr=ENV_ADDR)



while True:
    print("waiting for obs")
    obs_np = udp.recv_numpy(bufsize=8192, dtype=np.float32, timeout=None).reshape(4, 123)

    obs = torch.tensor(obs_np).to("cuda:0")
    acs = model.actor.forward(obs)

    print(acs.shape)

    acs_np = acs.cpu().detach().numpy().flatten()

    udp.send(acs_np)


