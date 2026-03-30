'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Mic, MicOff, Video, VideoOff } from 'lucide-react';
import { Room, RoomEvent } from 'livekit-client';
import { toast } from 'sonner';

interface RoomControlsProps {
  room: Room | null;
  onMuteChange?: (muted: boolean) => void;
  onVideoChange?: (enabled: boolean) => void;
  /** High-contrast overlay style for use on top of video */
  variant?: 'default' | 'floating';
}

export function RoomControls({ room, onMuteChange, onVideoChange, variant = 'default' }: RoomControlsProps) {
  const [isMuted, setIsMuted] = useState(false);
  const [isVideoEnabled, setIsVideoEnabled] = useState(true);
  const [isLoading, setIsLoading] = useState(false);

  // Sync with LiveKit local participant (do not use isSubscribed for local tracks — it breaks mute UI)
  useEffect(() => {
    if (!room) return;

    const updateStates = () => {
      const lp = room.localParticipant;
      setIsMuted(!lp.isMicrophoneEnabled);
      setIsVideoEnabled(lp.isCameraEnabled);
    };

    updateStates();

    room.on(RoomEvent.TrackPublished, updateStates);
    room.on(RoomEvent.TrackUnpublished, updateStates);
    room.on(RoomEvent.TrackMuted, updateStates);
    room.on(RoomEvent.TrackUnmuted, updateStates);
    room.on(RoomEvent.LocalTrackPublished, updateStates);
    room.on(RoomEvent.LocalTrackUnpublished, updateStates);

    return () => {
      room.off(RoomEvent.TrackPublished, updateStates);
      room.off(RoomEvent.TrackUnpublished, updateStates);
      room.off(RoomEvent.TrackMuted, updateStates);
      room.off(RoomEvent.TrackUnmuted, updateStates);
      room.off(RoomEvent.LocalTrackPublished, updateStates);
      room.off(RoomEvent.LocalTrackUnpublished, updateStates);
    };
  }, [room]);

  const toggleMute = async () => {
    console.log('toggleMute called', { room: !!room, roomState: room?.state });
    
    if (!room) {
      toast.error('Room not available');
      return;
    }

    // Only allow if room is connected
    if (room.state !== 'connected') {
      toast.error(`Room is ${room.state}. Please wait for connection.`);
      return;
    }

    setIsLoading(true);
    try {
      const localParticipant = room.localParticipant;
      const isCurrentlyMuted = !localParticipant.isMicrophoneEnabled;

      console.log('Mute state:', { isCurrentlyMuted });

      if (isCurrentlyMuted) {
        await localParticipant.setMicrophoneEnabled(true);
        setIsMuted(false);
        onMuteChange?.(false);
        toast.success('Microphone enabled');
      } else {
        await localParticipant.setMicrophoneEnabled(false);
        setIsMuted(true);
        onMuteChange?.(true);
        toast.success('Microphone muted');
      }
    } catch (error) {
      console.error('Failed to toggle mute:', error);
      toast.error(`Failed to toggle microphone: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleVideo = async () => {
    console.log('toggleVideo called', { room: !!room, roomState: room?.state });
    
    if (!room) {
      toast.error('Room not available');
      return;
    }

    // Only allow if room is connected
    if (room.state !== 'connected') {
      toast.error(`Room is ${room.state}. Please wait for connection.`);
      return;
    }

    setIsLoading(true);
    try {
      const localParticipant = room.localParticipant;
      const isCurrentlyEnabled = localParticipant.isCameraEnabled;

      console.log('Video state:', { isCurrentlyEnabled });

      if (isCurrentlyEnabled) {
        await localParticipant.setCameraEnabled(false);
        setIsVideoEnabled(false);
        onVideoChange?.(false);
        toast.success('Camera disabled');
      } else {
        await localParticipant.setCameraEnabled(true);
        setIsVideoEnabled(true);
        onVideoChange?.(true);
        toast.success('Camera enabled');
      }
    } catch (error) {
      console.error('Failed to toggle video:', error);
      toast.error(`Failed to toggle camera: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsLoading(false);
    }
  };

  const isRoomReady = room && room.state !== 'disconnected';

  const floating = variant === 'floating';

  const micBtnClass = floating
    ? `h-12 w-12 rounded-full shrink-0 border-2 border-white/40 shadow-lg ${
        isMuted
          ? 'bg-red-600 text-white hover:bg-red-700'
          : 'bg-zinc-900/95 text-white hover:bg-zinc-800'
      }`
    : '';

  const camBtnClass = floating
    ? `h-12 w-12 rounded-full shrink-0 border-2 border-white/40 shadow-lg ${
        isVideoEnabled
          ? 'bg-zinc-900/95 text-white hover:bg-zinc-800'
          : 'bg-amber-700 text-white hover:bg-amber-800'
      }`
    : `rounded-full h-11 w-11 shrink-0`;

  return (
    <div
      className={
        floating
          ? 'flex items-center justify-center gap-3 p-1'
          : 'flex items-center justify-center gap-3 p-2'
      }
      role="toolbar"
      aria-label="Microphone and camera"
    >
      <Button
        variant={floating ? 'ghost' : isMuted ? 'destructive' : 'default'}
        size="icon"
        onClick={toggleMute}
        className={floating ? micBtnClass : `rounded-full h-11 w-11 ${isMuted ? 'bg-destructive hover:bg-destructive/90' : ''}`}
        disabled={!isRoomReady || isLoading}
        title={!isRoomReady ? 'Waiting for room connection...' : isMuted ? 'Unmute microphone' : 'Mute microphone'}
      >
        {isMuted ? <MicOff className="h-5 w-5" /> : <Mic className="h-5 w-5" />}
      </Button>

      <Button
        variant={floating ? 'ghost' : isVideoEnabled ? 'default' : 'secondary'}
        size="icon"
        onClick={toggleVideo}
        className={floating ? camBtnClass : `rounded-full h-11 w-11`}
        disabled={!isRoomReady || isLoading}
        title={!isRoomReady ? 'Waiting for room connection...' : isVideoEnabled ? 'Turn camera off' : 'Turn camera on'}
      >
        {isVideoEnabled ? <Video className="h-5 w-5" /> : <VideoOff className="h-5 w-5" />}
      </Button>
    </div>
  );
}

