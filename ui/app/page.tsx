'use client'
import { useRouter } from 'next/navigation'
import { useEffect, useState, useRef } from 'react'

// Add interface for type safety
interface Detection {
  track_id: number;
  bbox: number[];
  confidence: number;
  class: number;
}

export default function Home() {
  const [detections, setDetections] = useState<Detection[]>([])
  const [selectedPerson, setSelectedPerson] = useState<number | null>(null)
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 })
  const imgRef = useRef<HTMLImageElement>(null)
  const router = useRouter()

  useEffect(() => {
    fetch('http://localhost:8000/yolo/predict', { method: 'GET' })
      .then(res => res.json())
      .then(setDetections)
  }, [])

  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget
    setImageSize({
      width: img.naturalWidth,
      height: img.naturalHeight
    })
  }

  const handleBoxClick = (trackId: number) => {
    setSelectedPerson(trackId)
    router.push(`/track?personId=${trackId}`)
  }

  function calculateBoxPosition(bbox: number[]) {
    if (!imgRef.current || !imageSize.width || !imageSize.height) {
      return { left: 0, top: 0, width: 0, height: 0 }
    }
  
    const imageElement = imgRef.current
    const displayedWidth = imageElement.offsetWidth
    const displayedHeight = imageElement.offsetHeight
  
    if (!displayedWidth || !displayedHeight) {
      return { left: 0, top: 0, width: 0, height: 0 }
    }
  
    const scaleX = displayedWidth / imageSize.width
    const scaleY = displayedHeight / imageSize.height
  
    const [x1, y1, x2, y2] = bbox
  
    return {
      left: Math.round(x1 * scaleX),
      top: Math.round(y1 * scaleY),
      width: Math.round((x2 - x1) * scaleX),
      height: Math.round((y2 - y1) * scaleY),
    }
  }

  return (
    <div className="w-full h-screen flex items-center justify-center p-4">
      <div className="inline-block relative">
        <img 
          ref={imgRef}
          src="http://localhost:8000/static/sample.jpg"
          alt="Scene"
          style={{
            height: '90vh',
            width: 'auto',
            objectFit: 'contain'
          }}
          onLoad={handleImageLoad}
        />
        {detections.map(({ track_id, bbox, confidence }) => {
          const { left, top, width, height } = calculateBoxPosition(bbox)
          return (
            <div
              key={track_id}
              onClick={() => handleBoxClick(track_id)}
              className={`absolute border-2 border-blue-500 cursor-pointer
                hover:bg-blue-500/20 transition-colors
                ${selectedPerson && selectedPerson !== track_id ? 'hidden' : ''}`}
              style={{
                left: `${left}px`,
                top: `${top}px`,
                width: `${width}px`,
                height: `${height}px`,
              }}
            >
              {/* Optional: Show confidence score */}
              <span className="absolute -top-6 left-0 bg-blue-500 text-white px-2 py-1 text-xs rounded">
                {track_id}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}