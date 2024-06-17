import argparse
import functools
import io
import json
import base64
import time

import torch
from aiohttp import web
from PIL import Image,ImageDraw

from rotate_captcha_crack.common import device
from rotate_captcha_crack.const import DEFAULT_CLS_NUM
from rotate_captcha_crack.logging import RCCLogger
from rotate_captcha_crack.model import RotNetR, WhereIsMyModel
from rotate_captcha_crack.utils import process_captcha

logger = RCCLogger()
routes = web.RouteTableDef()

dumps = functools.partial(json.dumps, separators=(',', ':'))

parser = argparse.ArgumentParser()
parser.add_argument("--index", "-i", type=int, default=-1, help="Use which index")
opts = parser.parse_args()

model = RotNetR(cls_num=DEFAULT_CLS_NUM, train=False)
model_path = WhereIsMyModel(model).with_index(opts.index).model_dir / "best.pth"
model.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu')))
model = model.to(device=device)
model.eval()




@routes.post('/')
async def hello(request: web.Request):
    resp = {'err': {'code': 0, 'msg': 'success'}}

    try:
        data = await request.post()
        img_base64 = data.get('img', "Anonymous").replace(" ","+")

        img_bytes = base64.b64decode(img_base64)
        img = Image.open(io.BytesIO(img_bytes))

        circle_image = Image.new("RGBA", img.size)
        draw = ImageDraw.Draw(circle_image)
        draw.ellipse([(0, 0), img.size], fill=(255, 255, 255))
        result = Image.new("RGBA", img.size)
        result.paste(img, mask=circle_image)
        # 保存截取后的图像
        result.save("C:/Users/admin/rotate-captcha-crack/datasets/"+time.strftime('%Y-%m-%d %H-%M-%S.png', time.localtime(time.time())))




        with torch.no_grad():
            img_ts = process_captcha(result)
            img_ts = img_ts.to(device=device)
            predict = model.predict(img_ts) * 360
            resp['pred'] = predict

    except Exception as err:
        resp['err']['code'] = 0x0001
        resp['err']['msg'] = str(err)
        # resp['err']['img'] = img_base64
        return web.json_response(resp, status=400, dumps=dumps)

    return web.json_response(resp, dumps=dumps)


app = web.Application()
app.add_routes(routes)
web.run_app(app, port=4396, access_log_format='%a "%r" %s %b', access_log=logger)
