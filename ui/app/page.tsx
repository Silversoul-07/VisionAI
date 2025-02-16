'use client'
import { useRouter } from 'next/navigation'
import { useEffect, useState, useRef } from 'react'

export default function Home() {
  const [detections, setDetections] = useState([])
  const [selectedPerson, setSelectedPerson] = useState(null)
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 })
  const imgRef = useRef(null)
  const router = useRouter()

  useEffect(() => {
    fetch('http://localhost:8000/yolo/predict', { method: 'GET' })
      .then(res => res.json())
      .then(setDetections)
  }, [])

  const handleImageLoad = (e) => {
    const img = e.target
    setImageSize({
      width: img.naturalWidth,
      height: img.naturalHeight
    })
  }

  const handleBoxClick = (personId) => {
    setSelectedPerson(personId)
    router.push(`/track?personId=${personId}`)
  }

  function calculateBoxPosition(bbox) {
    if (!imgRef.current || !imageSize.width || !imageSize.height) {
      return { left: 0, top: 0, width: 0, height: 0 }
    }
  
    const imageElement = imgRef.current
    const displayedWidth = imageElement.offsetWidth
    const displayedHeight = imageElement.offsetHeight
  
    // Ensure we have valid dimensions
    if (!displayedWidth || !displayedHeight) {
      return { left: 0, top: 0, width: 0, height: 0 }
    }
  
    // Calculate scale based on the original and displayed dimensions
    const scaleX = displayedWidth / imageSize.width
    const scaleY = displayedHeight / imageSize.height
  
    // Handle absolute pixel coordinates [x1, y1, x2, y2]
    const [x1, y1, x2, y2] = bbox
  
    // Convert absolute coordinates to displayed coordinates
    const res = {
      left: Math.round(x1 * scaleX),
      top: Math.round(y1 * scaleY),
      width: Math.round((x2 - x1) * scaleX),
      height: Math.round((y2 - y1) * scaleY),
    }
    console.log(res)
    return res
  }

  return (
    <div className="w-full h-screen flex items-center justify-center p-4">
      {/* Change made here: use inline-block so the container shrinks to fit the image */}
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
        {detections.map(({ id, bbox }) => {
          const { left, top, width, height } = calculateBoxPosition(bbox)
          return (
            <div
              key={id}
              onClick={() => handleBoxClick(id)}
              className={`absolute border-2 border-blue-500 cursor-pointer
                hover:bg-blue-500/20 transition-colors
                ${selectedPerson && selectedPerson !== id ? 'hidden' : ''}`}
              style={{
                left: `${left}px`,
                top: `${top}px`,
                width: `${width}px`,
                height: `${height}px`,
              }}
            />
          )
        })}
      </div>
    </div>
  )
}