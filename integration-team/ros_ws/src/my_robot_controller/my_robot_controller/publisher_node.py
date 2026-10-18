import rclpy
from rclpy.node import Node
from std_msgs.msg import String  # Change this if using a different message type


class GenericPublisher(Node):
    def __init__(self):
        super().__init__('generic_publisher')
        
        # --- Configuration ---
        topic_name = 'chatter'
        publish_frequency = 1.0  # Time in seconds between messages
        
        # Create publisher (Message Type, Topic Name, Queue Size)
        self.publisher_ = self.create_publisher(String, topic_name, 10)
        
        # Create a timer to trigger periodic publishing
        self.timer = self.create_timer(publish_frequency, self.timer_callback)
        self.counter = 0
        
        self.get_logger().info(
            f'Publisher initialized. Broadcasting on "{topic_name}" every {publish_frequency}s.')

    def timer_callback(self):
        msg = String()
        
        # --- Data Payload Logic ---
        msg.data = f'Data packet index: {self.counter}'
        
        # Publish the message
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        
        self.counter += 1

def main(args=None):
    rclpy.init(args=args)
    node = GenericPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()