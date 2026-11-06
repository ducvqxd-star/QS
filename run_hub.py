import socket
import uvicorn
import webbrowser
import os
import sys

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

if __name__ == "__main__":
    local_ip = get_local_ip()
    port = 8080
    
    print("\n" + "="*65)
    print("      HE THONG DINH GIA & DU TOAN NOI THAT QS STUDIO (HUB)")
    print("="*65)
    print(f"[*] May chu cua ban truy cap tai : http://localhost:{port}")
    print(f"[*] Nhan vien cung mang Wi-Fi vao: http://{local_ip}:{port}")
    print("="*65)
    print("May chu dang khoi chay, vui long giu nguyen cua so nay...")
    
    try:
        webbrowser.open(f"http://localhost:{port}")
    except Exception:
        pass
        
    from server import app
    uvicorn.run(app, host="0.0.0.0", port=port)
