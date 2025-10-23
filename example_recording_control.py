"""
Example: Recording control with DLCClient and MyProcessor_socket

This demonstrates:
1. Starting a processor
2. Connecting a client
3. Controlling recording (start/stop/save) from the client
4. Session name management
"""

import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mouse_ar.ctrl.dlc_client import DLCClient


def example_recording_workflow():
    """Complete workflow: processor + client with recording control."""
    
    print("\n" + "="*70)
    print("EXAMPLE: Recording Control Workflow")
    print("="*70)
    
    # NOTE: This example assumes MyProcessor_socket is already running
    # Start it separately with:
    #   from dlc_processor_socket import MyProcessor_socket
    #   processor = MyProcessor_socket(bind=("localhost", 6000))
    #   # Then run DLCLive with this processor
    
    print("\n[CLIENT] Connecting to processor at localhost:6000...")
    client = DLCClient(address=("localhost", 6000))
    
    try:
        # Start the client (connects and begins receiving data)
        client.start()
        print("[CLIENT] Connected!")
        time.sleep(0.5)  # Wait for connection to stabilize
        
        # Set session name
        print("\n[CLIENT] Setting session name to 'experiment_001'...")
        client.set_session_name("experiment_001")
        time.sleep(0.2)
        
        # Start recording
        print("[CLIENT] Starting recording (clears processor data queues)...")
        client.start_recording()
        time.sleep(0.2)
        
        # Receive some data
        print("\n[CLIENT] Receiving data for 5 seconds...")
        for i in range(5):
            data = client.read()
            if data:
                vals = data["vals"]
                print(f"  t={vals[0]:.2f}, x={vals[1]:.1f}, y={vals[2]:.1f}, "
                      f"heading={vals[3]:.1f}°, head_angle={vals[4]:.2f}rad")
            time.sleep(1.0)
        
        # Stop recording
        print("\n[CLIENT] Stopping recording...")
        client.stop_recording()
        time.sleep(0.2)
        
        # Trigger save
        print("[CLIENT] Triggering save on processor...")
        client.trigger_save()  # Uses processor's default filename
        # OR specify custom filename:
        # client.trigger_save(filename="my_custom_data.pkl")
        time.sleep(0.5)
        
        print("\n[CLIENT] ✓ Workflow complete!")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        print("\nMake sure MyProcessor_socket is running!")
        print("Example:")
        print("  from dlc_processor_socket import MyProcessor_socket")
        print("  processor = MyProcessor_socket()")
        print("  # Then run DLCLive with this processor")
    
    finally:
        print("\n[CLIENT] Closing connection...")
        client.close()


def example_multiple_sessions():
    """Example: Recording multiple sessions with the same processor."""
    
    print("\n" + "="*70)
    print("EXAMPLE: Multiple Sessions")
    print("="*70)
    
    client = DLCClient(address=("localhost", 6000))
    
    try:
        client.start()
        print("[CLIENT] Connected!")
        time.sleep(0.5)
        
        # Session 1
        print("\n--- SESSION 1 ---")
        client.set_session_name("trial_001")
        client.start_recording()
        print("Recording session 'trial_001' for 3 seconds...")
        time.sleep(3.0)
        client.stop_recording()
        client.trigger_save()  # Saves as "trial_001_dlc_processor_data.pkl"
        print("Session 1 saved!")
        
        time.sleep(1.0)
        
        # Session 2
        print("\n--- SESSION 2 ---")
        client.set_session_name("trial_002")
        client.start_recording()
        print("Recording session 'trial_002' for 3 seconds...")
        time.sleep(3.0)
        client.stop_recording()
        client.trigger_save()  # Saves as "trial_002_dlc_processor_data.pkl"
        print("Session 2 saved!")
        
        print("\n✓ Multiple sessions recorded successfully!")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
    
    finally:
        client.close()


def example_command_api():
    """Example: Using the low-level command API."""
    
    print("\n" + "="*70)
    print("EXAMPLE: Low-level Command API")
    print("="*70)
    
    client = DLCClient(address=("localhost", 6000))
    
    try:
        client.start()
        time.sleep(0.5)
        
        # Using send_command directly
        print("\n[CLIENT] Using send_command()...")
        
        # Set session name
        client.send_command("set_session_name", session_name="custom_session")
        print("  ✓ Sent: set_session_name")
        time.sleep(0.2)
        
        # Start recording
        client.send_command("start_recording")
        print("  ✓ Sent: start_recording")
        time.sleep(2.0)
        
        # Stop recording
        client.send_command("stop_recording")
        print("  ✓ Sent: stop_recording")
        time.sleep(0.2)
        
        # Save with custom filename
        client.send_command("save", filename="my_data.pkl")
        print("  ✓ Sent: save")
        time.sleep(0.5)
        
        print("\n✓ Commands sent successfully!")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
    
    finally:
        client.close()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DLC PROCESSOR RECORDING CONTROL EXAMPLES")
    print("="*70)
    print("\nNOTE: These examples require MyProcessor_socket to be running.")
    print("Start it separately before running these examples.")
    print("="*70)
    
    # Uncomment the example you want to run:
    
    # example_recording_workflow()
    # example_multiple_sessions()
    # example_command_api()
    
    print("\nUncomment an example in the script to run it.")
