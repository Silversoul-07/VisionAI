
// app/video/page.jsx
'use client'
import { useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'next/navigation'

export default function VideoStream() {
    const personId = useSearchParams().get('personId')
  const [videoData, setVideoData] = useState(null)
  const [detections, setDetections] = useState([])
  const canvasRef = useRef(null)

  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8000/track/${personId}`)
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      setVideoData(data.frame)
      setDetections(data.detections)
    }

    return () => ws.close()
  }, [])

  useEffect(() => {
    if (!videoData || !canvasRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    
    const image = new Image()
    image.src = `data:image/jpeg;base64,${videoData}`
    
    image.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height)
      
      detections.forEach(({ bbox }) => {
        ctx.strokeStyle = '#3B82F6'
        ctx.lineWidth = 2
        ctx.strokeRect(
          bbox[0] * canvas.width / 100,
          bbox[1] * canvas.height / 100,
          bbox[2] * canvas.width / 100,
          bbox[3] * canvas.height / 100
        )
      })
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