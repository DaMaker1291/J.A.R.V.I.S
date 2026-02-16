import { useEffect, useState } from 'react';
import io, { Socket } from 'socket.io-client';

const useSocket = () => {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [status, setStatus] = useState<{
    status: string;
    mood: string;
    threat_level: number;
  }>({
    status: 'idle',
    mood: 'blue',
    threat_level: 0.0,
  });

  useEffect(() => {
    const socketInstance = io('http://localhost:5000');

    socketInstance.on('connect', () => {
      setIsConnected(true);
    });

    socketInstance.on('disconnect', () => {
      setIsConnected(false);
    });

    socketInstance.on('status_update', (data) => {
      setStatus(data);
    });

    setSocket(socketInstance);

    return () => {
      socketInstance.disconnect();
    };
  }, []);

  return { socket, isConnected, status };
};

export default useSocket;
