

from flask import Flask
from flask import request, jsonify
from API import aigen

app = Flask(__name__)

user = {}
@app.route('/api/<id>',
           methods=['POST'])
def index(id):
    """
    接收POST请求，处理AI生成的内容。
    :param id: 请求的ID
    :return: JSON格式的响应
    """
    if id not in user:
        user.setdefault(id, {"point"})
