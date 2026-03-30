'use client';

import { useEffect, useState, useRef } from 'react';
import { Room } from 'livekit-client';
import { Card, CardContent } from '@/components/ui/card';
import { User, UserCheck } from 'lucide-react';

interface TranscriptionDisplayProps {
  room: Room | null;
}

interface TranscriptionMessage {
  id: string;
  text: string;
  speaker: string;
  timestamp: Date;
  isFinal: boolean;
}

export function TranscriptionDisplay({ room }: TranscriptionDisplayProps) {
  const [messages, setMessages] = useState<TranscriptionMessage[]>([]);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!room) return;

    console.log('Setting up transcription text stream handler for room:', room.name);

    // AgentSession STT outputs are sent via text streams, NOT RoomEvent.TranscriptionReceived
    // We must use registerTextStreamHandler with topic 'lk.transcription'
    const handler = async (reader: any, participantInfo?: any) => {
      try {
        const textData = await reader.readAll();
        const fullText = Array.isArray(textData) ? textData.join('') : textData;
        
        // Check if this is a final transcription
        const isFinal = reader.info?.attributes?.['lk.transcription_final'] === 'true';
        const segmentId = reader.info?.attributes?.['lk.segment_id'];
        const transcribedTrackId = reader.info?.attributes?.['lk.transcribed_track_id'];
        
        console.log('📝 Transcription received:', {
          text: fullText,
          isFinal,
          segmentId,
          participant: participantInfo?.identity || participantInfo?.name || 'Unknown',
          transcribedTrackId,
        });

        // Normalize speaker so local STT segments share one key (merge consecutive bubbles)
        const lp = room.localParticipant;
        const pid = participantInfo?.identity ?? '';
        const pname = participantInfo?.name ?? '';
        const isLocalSpeaker =
          !!lp &&
          (pid === lp.identity ||
            pid === lp.name ||
            pname === lp.name ||
            pname === lp.identity ||
            (pid && lp.identity && pid.toLowerCase() === lp.identity.toLowerCase()));

        let speakerName = participantInfo?.name || participantInfo?.identity || 'Unknown';
        if (isLocalSpeaker) {
          speakerName = lp!.name || lp!.identity || 'You';
        } else if (
          speakerName.toLowerCase().includes('agent') ||
          speakerName.toLowerCase().includes('ai') ||
          speakerName.toLowerCase().includes('interviewer')
        ) {
          speakerName = 'Interviewer';
        }

        setMessages((prev) => {
          let newMessages = [...prev];

          if (isFinal) {
            // Drop interim bubble for this speaker when final arrives
            newMessages = newMessages.filter(
              (m) => !(m.speaker === speakerName && !m.isFinal)
            );

            const last = newMessages[newMessages.length - 1];
            const chunk = typeof fullText === 'string' ? fullText.trim() : String(fullText).trim();
            if (!chunk) {
              return newMessages.slice(-100);
            }

            const canMerge =
              last &&
              last.isFinal &&
              last.speaker === speakerName;

            if (canMerge) {
              const prevText = last.text.trim();
              newMessages[newMessages.length - 1] = {
                ...last,
                text: [prevText, chunk].filter(Boolean).join(' '),
                timestamp: new Date(),
              };
            } else {
              const uniqueId = `${segmentId || 'final'}-${speakerName}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
              newMessages.push({
                id: uniqueId,
                text: chunk,
                speaker: speakerName,
                timestamp: new Date(),
                isFinal: true,
              });
            }
          } else {
            // One interim row per speaker (avoids stacked "typing..." duplicates)
            const interimKey = `interim-${speakerName}`;
            const existingIndex = newMessages.findIndex(
              (m) => !m.isFinal && m.speaker === speakerName
            );
            if (existingIndex >= 0) {
              newMessages[existingIndex] = {
                ...newMessages[existingIndex],
                id: interimKey,
                text: fullText,
                timestamp: new Date(),
              };
            } else {
              newMessages.push({
                id: interimKey,
                text: fullText,
                speaker: speakerName,
                timestamp: new Date(),
                isFinal: false,
              });
            }
          }

          return newMessages.slice(-100);
        });
      } catch (error) {
        console.error('Error reading transcription text stream:', error);
      }
    };

    // Register text stream handler for AgentSession transcriptions
    // Note: registerTextStreamHandler doesn't return a Promise, so no .catch()
    try {
      room.registerTextStreamHandler('lk.transcription', handler);
      console.log('✅ Transcription text stream handler registered for topic: lk.transcription');
    } catch (error) {
      console.error('Failed to register transcription text stream handler:', error);
    }

    return () => {
      try {
        room.unregisterTextStreamHandler('lk.transcription');
      } catch (error) {
        console.warn('Failed to unregister transcription handler:', error);
      }
    };
  }, [room]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <Card className="flex h-full min-h-0 flex-1 flex-col overflow-hidden border-0 shadow-none">
      <CardContent className="flex min-h-0 flex-1 flex-col gap-0 p-0">
        <h3 className="shrink-0 px-1 pb-2 text-sm font-semibold">Conversation</h3>
        <div
          className="min-h-0 flex-1 flex flex-col gap-4 overflow-y-auto overflow-x-hidden pr-1 pb-1"
          ref={scrollAreaRef}
        >
          {messages.length === 0 ? (
            <p className="py-6 text-center text-sm text-muted-foreground">
              Waiting for conversation to start...
            </p>
          ) : (
            messages.map((message) => {
              const isInterviewer = message.speaker === 'Interviewer' ||
                                   message.speaker.toLowerCase().includes('interviewer') ||
                                   message.speaker.toLowerCase().includes('agent') ||
                                   message.speaker.toLowerCase().includes('ai');
              const isUser = message.speaker === room?.localParticipant?.name || 
                           message.speaker === room?.localParticipant?.identity;
              
              const displayName = isUser ? 'You' : (isInterviewer ? 'Interviewer' : message.speaker);
              
              return (
                <div
                  key={message.id}
                  className={`flex shrink-0 items-start gap-3 rounded-lg border px-3 py-3 ${
                    isInterviewer 
                      ? 'border-primary/20 bg-primary/5' 
                      : isUser
                      ? 'border-muted bg-muted/40'
                      : 'border-border bg-background'
                  } ${!message.isFinal ? 'opacity-80' : ''}`}
                >
                  <div className={`mt-0.5 shrink-0 ${
                    isInterviewer ? 'text-primary' : 'text-muted-foreground'
                  }`}>
                    {isInterviewer ? (
                      <UserCheck className="h-4 w-4" />
                    ) : (
                      <User className="h-4 w-4" />
                    )}
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="mb-1 flex flex-wrap items-center gap-x-2 gap-y-0.5">
                      <span className={`text-xs font-semibold ${
                        isInterviewer ? 'text-primary' : 'text-foreground'
                      }`}>
                        {displayName}
                      </span>
                      {!message.isFinal && (
                        <span className="text-xs italic text-muted-foreground">(typing…)</span>
                      )}
                    </div>
                    <p className={`break-words text-sm leading-relaxed ${
                      !message.isFinal ? 'italic' : ''
                    }`}>
                      {message.text}
                    </p>
                    <p className="mt-1.5 text-xs text-muted-foreground">
                      {message.timestamp.toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </CardContent>
    </Card>
  );
}

