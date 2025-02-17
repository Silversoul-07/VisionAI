'use client'
import { useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'next/navigation'

interface Detection {
  person_id: string;
  bbox: number[];
  confidence: number;
}

interface WebSocketMessage {
  detections: Detection;
  timestamp: string;
  result_image: string;
}

export default function VideoStream() {
  const personId = useSearchParams().get('personId')
  const [videoData, setVideoData] = useState<string | null>(null)
  const [detections, setDetections] = useState<Detection | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8000/ws/track/${personId}`)
    
    ws.onmessage = (event) => {
      const data: WebSocketMessage = JSON.parse(event.data)
      setVideoData(data.result_image)
      setDetections(data.detections)
    }

    return () => ws.close()
  }, [personId])

  useEffect(() => {
    if (!videoData || !canvasRef.current || !detections) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const image = new Image()
    image.src = `${videoData}`
    
    image.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height)
      
      // Draw single detection bbox
      if (detections.bbox) {
        ctx.strokeStyle = '#3B82F6'
        ctx.lineWidth = 2
        ctx.strokeRect(
          detections.bbox[0],
          detections.bbox[1],
          detections.bbox[2] - detections.bbox[0],
          detections.bbox[3] - detections.bbox[1]
        )

        // Optional: Draw confidence score
        if (detections.confidence) {
          ctx.fillStyle = '#3B82F6'
          ctx.font = '14px Arial'
          ctx.fillText(
            `${Math.round(detections.confidence * 100)}%`,
            detections.bbox[0],
            detections.bbox[1] - 5
          )
        }
      }
    }
  }, [videoData, detections])

  return (
    <div className="w-full max-w-6xl mx-auto p-4">
      <canvas
        ref={canvasRef}
        className="w-full h-auto"
        width={1280}
        height={720}
      />
    </div>
  )
}