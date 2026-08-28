export default function ChunkGradeCards({ chunks, grades }) {
  if (!chunks || chunks.length === 0) return null

  const relevanceById = Object.fromEntries((grades || []).map(g => [g.chunkId, g.relevant]))

  return (
    <div className="chunks-container">
      {chunks.map((chunk) => {
        const relevant = relevanceById[chunk.id]
        const cls = relevant === false ? 'irrelevant' : 'relevant'
        return (
          <div key={chunk.id} className={`chunk-item graded ${cls}`}>
            <div className="chunk-header">
              <span className="chunk-index">Rank #{chunk.rank}, Chunk #{chunk.id}</span>
              <span className={`grade-tag ${cls}`}>{relevant === false ? '✕ irrelevant' : '✓ relevant'}</span>
            </div>
            {chunk.text}
          </div>
        )
      })}
    </div>
  )
}
