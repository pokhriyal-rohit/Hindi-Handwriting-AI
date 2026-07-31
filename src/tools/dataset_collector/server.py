import os
import json
from http.server import SimpleHTTPRequestHandler, HTTPServer

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
                writer_id = data.get("writer_id", "writer_unknown")
                
                # Hierarchical storage: data/raw/custom_hindi/<writer_id>/<word>/sample_xxx.json
                writer_dir = os.path.join(DATA_DIR, writer_id, word)
                os.makedirs(writer_dir, exist_ok=True)
                
                # Metadata dump (just dumping the latest, effectively updating it per writer)
                metadata_path = os.path.join(DATA_DIR, writer_id, "metadata.json")
                if not os.path.exists(metadata_path):
                    with open(metadata_path, "w", encoding="utf-8") as mf:
                        json.dump({
                            "writer_id": writer_id,
                            "device": data.get("device"),
                            "script": data.get("script")
                        }, mf, indent=2)
                
                # Find next ID for this word/writer combo
                existing = [f for f in os.listdir(writer_dir) if f.startswith("sample_")]
                next_id = len(existing) + 1
                
                filename = os.path.join(writer_dir, f"sample_{next_id:03d}.json")
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
    
    web_dir = os.path.dirname(__file__)
    os.chdir(web_dir)
    
    httpd = server_class(server_address, handler_class)
    print(f"Starting Data Collector Server at http://localhost:{port}")
    print(f"Data will be saved hierarchically to: {DATA_DIR}/<writer_id>/<word>/")
    httpd.serve_forever()

if __name__ == '__main__':
    run()
