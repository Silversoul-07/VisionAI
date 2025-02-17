'use client'
import { useEffect, useRef, useState } from 'react'

export default function VideoStream() {
  const video1Ref = useRef(null)
  const video2Ref = useRef(null)
  const canvas1Ref = useRef(null)
  const canvas2Ref = useRef(null)
  const [trackingId, setTrackingId] = useState('')
  const [isPlaying, setIsPlaying] = useState(false)
  const detectionInterval = useRef(null)
  const API_BASE = 'http://localhost:8000' // FastAPI backend URL

  // Load videos from backend
  useEffect(() => {
    const loadVideos = async () => {
      try {
        const response = await fetch(`${API_BASE}/yolo/predict`)
        const data = await response.json()
        
        video1Ref.current.src = `${API_BASE}${data.video1_url}`
        video2Ref.current.src = `${API_BASE}${data.video2_url}`
      } catch (error) {
        console.error('Error loading videos:', error)
      }
    }

    loadVideos()
  }, [])

  // Fetch detections from backend
  const fetchDetections = async (videoNumber) => {
    try {
      const response = await fetch(`${API_BASE}/yolo/detections/${videoNumber}`)
      const detections = await response.json()
      return detections
    } catch (error) {
      console.error('Error fetching detections:', error)
      return []
    }
  }

  // Draw detections on canvas
  const drawDetections = (canvasRef, detections, isVideo2 = false) => {
    const ctx = canvasRef.current.getContext('2d')
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)

    detections.forEach(detection => {
      const { x, y, width, height, person_id } = detection
      
      // Draw all detections for video1
      if (!isVideo2) {
        ctx.strokeStyle = '#FF0000'
        ctx.lineWidth = 2
        ctx.strokeRect(x, y, width, height)
        ctx.fillStyle = '#FF0000'
        ctx.fillText(`ID: ${person_id}`, x, y > 10 ? y - 5 : 10)
      }
      
      // Draw only tracked person for video2
      if (isVideo2 && person_id === trackingId) {
        ctx.strokeStyle = '#00FF00'
        ctx.lineWidth = 2
        ctx.strokeRect(x, y, width, height)
        ctx.fillStyle = '#00FF00'
        ctx.fillText(`Tracked ID: ${person_id}`, x, y > 10 ? y - 5 : 10)
      }
    })
  }

  // Start detection polling when playing
  useEffect(() => {
    if (isPlaying) {
      detectionInterval.current = setInterval(async () => {
        const video1Detections = await fetchDetections(1)
        const video2Detections = await fetchDetections(2)
        
        drawDetections(canvas1Ref, video1Detections)
        drawDetections(canvas2Ref, video2Detections, true)
      }, 100) // Update every 100ms
    } else {
      clearInterval(detectionInterval.current)
    }

    return () => clearInterval(detectionInterval.current)
  }, [isPlaying, trackingId])

  // Handle track button click
  const handleTrackPerson = () => {
    if (trackingId) {
      // Send tracking ID to backend
      fetch(`${API_BASE}/yolo/track`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ person_id: trackingId })
      })
    }
  }

  // Sync video play/pause
  const handlePlayPause = () => {
    setIsPlaying(!isPlaying)
    if (isPlaying) {
      video1Ref.current.pause()
      video2Ref.current.pause()
    } else {
      video1Ref.current.play()
      video2Ref.current.play()
    }
  }

  return (
    <div className="flex flex-col items-center p-4">
      <div className="flex gap-4 mb-4">
        {/* Video 1 */}
        <div className="relative">
          <video ref={video1Ref} className="w-[640px] h-[360px] border" />
          <canvas
            ref={canvas1Ref}
            className="absolute top-0 left-0 w-full h-full pointer-events-none"
            width={1280}
            height={720}
          />
        </div>

        {/* Video 2 */}
        <div className="relative">
          <video ref={video2Ref} className="w-[640px] h-[360px] border" />
          <canvas
            ref={canvas2Ref}
            className="absolute top-0 left-0 w-full h-full pointer-events-none"
            width={1280}
            height={720}
          />
        </div>
      </div>

      <div className="flex gap-4 mb-4">
        <input
          type="text"
          placeholder="Enter Person ID to track"
          value={trackingId}
          onChange={(e) => setTrackingId(e.target.value)}
          className="px-4 py-2 border rounded"
        />
        <button
          onClick={handleTrackPerson}
          className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
        >
          Track Person
        </button>
      </div>

      <button
        onClick={handlePlayPause}
        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
      >
        {isPlaying ? 'Pause' : 'Play'}
      </button>
    </div>
  )
}