import os
import json
from http.server import SimpleHTTPRequestHandler, HTTPServer
import urllib.parse

PORT = 8080
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw", "custom_hindi"))

class CollectorHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/save':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                word = data.get("word", "unknown")
                
                os.makedirs(DATA_DIR, exist_ok=True)
                
                # Find next ID for this word
                existing = [f for f in os.listdir(DATA_DIR) if f.startswith(f"{word}_")]
                next_id = len(existing) + 1
                
                filename = os.path.join(DATA_DIR, f"{word}_{next_id:03d}.json")
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "file": filename}).encode('utf-8'))
                print(f"Saved: {filename}")
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                print(f"Error saving: {e}")
        else:
            self.send_response(404)
            self.end_headers()

def run(server_class=HTTPServer, handler_class=CollectorHandler, port=PORT):
    server_address = ('', port)
    
    # Change directory to serve index.html
    web_dir = os.path.dirname(__file__)
    os.chdir(web_dir)
    
    httpd = server_class(server_address, handler_class)
    print(f"Starting Data Collector Server at http://localhost:{port}")
    print(f"Data will be saved to: {DATA_DIR}")
    httpd.serve_forever()

if __name__ == '__main__':
    run()
