#!/usr/bin/env python3
"""
Visage Startup Script
Face Recognition Photo Management System
"""

import os
import sys
import subprocess
import socket
import platform

def print_banner():
    """Print the Visage banner"""
    print()
    print(" ██╗   ██╗██╗███████╗ █████╗  ██████╗ ███████╗")
    print(" ██║   ██║██║██╔════╝██╔══██╗██╔════╝ ██╔════╝")
    print(" ██║   ██║██║███████╗███████║██║  ███╗█████╗  ")
    print(" ╚██╗ ██╔╝██║╚════██║██╔══██║██║   ██║██╔══╝  ")
    print("  ╚████╔╝ ██║███████║██║  ██║╚██████╔╝███████╗")
    print("   ╚═══╝  ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝")
    print()
    print(" Face Recognition Photo Management System")
    print(" ========================================")
    print()

def get_local_ip():
    """Get the local IP address"""
    try:
        # Connect to a remote server to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "0.0.0.0"

def check_python():
    """Check if Python is available"""
    try:
        result = subprocess.run([sys.executable, "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Python found: {result.stdout.strip()}")
            return True
    except Exception:
        pass
    
    print("❌ Error: Python is not available")
    print("   Please install Python from https://python.org")
    return False

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import flask
        print("✅ Flask found")
        return True
    except ImportError:
        print("❌ Flask not found. Installing dependencies...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                         check=True)
            print("✅ Dependencies installed")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False

def display_network_info(host, port):
    """Display network information"""
    print(f"🌐 Running in {'LOCAL' if host == '127.0.0.1' else 'NETWORK'} mode")
    print(f"   Port: {port}")
    
    if host == '127.0.0.1':
        print(f"   URL: http://localhost:{port}")
        print("   Accessible only on this computer")
    else:
        local_ip = get_local_ip()
        print(f"   Local URL: http://localhost:{port}")
        print(f"   Network URL: http://{local_ip}:{port}")
        print(f"   Other devices can access: http://{local_ip}:{port}")
    
    print("   Press Ctrl+C to stop the server")
    print("=" * 50)

def main():
    """Main startup function"""
    print_banner()
    
    # Check Python
    if not check_python():
        input("Press Enter to exit...")
        return
    
    # Check dependencies
    if not check_dependencies():
        input("Press Enter to exit...")
        return
    
    print()
    
    # Display menu
    print("🌐 Choose how to run Visage:")
    print()
    print("   [1] Local Only - Accessible only on this computer")
    print("   [2] Network Mode - Accessible on local network")
    print("   [3] Custom Port - Choose your own port")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-3): ").strip()
            
            if choice == "1":
                print()
                print("🏠 Starting Visage in LOCAL mode...")
                print("   Only accessible from this computer")
                print()
                display_network_info("127.0.0.1", 5000)
                subprocess.run([sys.executable, "app.py", "--host", "local"])
                break
                
            elif choice == "2":
                print()
                print("🌐 Starting Visage in NETWORK mode...")
                print("   Accessible from other devices on your network")
                print()
                display_network_info("0.0.0.0", 5000)
                subprocess.run([sys.executable, "app.py", "--host", "network"])
                break
                
            elif choice == "3":
                print()
                try:
                    port = input("Enter port number (default 5000): ").strip()
                    if not port:
                        port = 5000
                    else:
                        port = int(port)
                except ValueError:
                    print("❌ Invalid port number. Using default 5000.")
                    port = 5000
                
                print()
                print("🔧 Choose host mode:")
                print("   [1] Local Only")
                print("   [2] Network Mode")
                print()
                
                while True:
                    host_choice = input("Enter choice (1-2): ").strip()
                    if host_choice == "1":
                        print()
                        print(f"🏠 Starting Visage in LOCAL mode on port {port}...")
                        display_network_info("127.0.0.1", port)
                        subprocess.run([sys.executable, "app.py", "--host", "local", "--port", str(port)])
                        break
                    elif host_choice == "2":
                        print()
                        print(f"🌐 Starting Visage in NETWORK mode on port {port}...")
                        display_network_info("0.0.0.0", port)
                        subprocess.run([sys.executable, "app.py", "--host", "network", "--port", str(port)])
                        break
                    else:
                        print("❌ Invalid choice. Please enter 1 or 2.")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            break
    
    print("\n👋 Visage has stopped.")

if __name__ == "__main__":
    main()
