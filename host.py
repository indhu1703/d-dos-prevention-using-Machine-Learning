from http.server import SimpleHTTPRequestHandler, HTTPServer
import os

class MyHTTPRequestHandler(SimpleHTTPRequestHandler):
    blocked_ips = {'192.168.115.172', '192.168.1.100'} 
    
    def do_GET(self):
        print(self.client_address[0])
        if self.client_address[0] in self.blocked_ips:
            self.send_error(403, "Forbidden")
            return
        
        if self.path == '/':
            self.path = 'index.html'  
        
        
        return SimpleHTTPRequestHandler.do_GET(self)

def run(server_class=HTTPServer, handler_class=MyHTTPRequestHandler, port=5000, bind_address=''):
    server_address = (bind_address, port)   

    httpd = server_class(server_address, handler_class)
    print(f'Starting server on {bind_address}:{port}...')
    httpd.serve_forever()
    
if __name__ == "__main__":
    
    bind_address = '192.168.149.6'  
    run(bind_address=bind_address)
