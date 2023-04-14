importScripts('https://cdn.jsdelivr.net/npm/@xenova/transformers/dist/transformers.min.js');

function calculateCosineSimilarity(queryEmbedding, embedding) {
  let dotProduct = 0;
  let queryMagnitude = 0;
  let embeddingMagnitude = 0;
  let queryEmbeddingLength = queryEmbedding.length
  for (let i = 0; i < queryEmbeddingLength; i++) {
      dotProduct += queryEmbedding[i] * embedding[i];
      queryMagnitude += queryEmbedding[i] ** 2;
      embeddingMagnitude += embedding[i] ** 2;
  }
  return dotProduct / (Math.sqrt(queryMagnitude) * Math.sqrt(embeddingMagnitude));
}

function getSents(article){
  const segmenterEn = new Intl.Segmenter('en', { granularity: 'sentence'})
  return [...segmenterEn.segment(article)]
}

self.addEventListener('message', async (e) => {
    const prompt = e.data[0]
    const task = e.data[1]
    const model = e.data[2]
    const then = e.data[3]

    if(task==='search_article') {
      try {
        let pipe1 = await pipeline("embeddings", "sentence-transformers/all-MiniLM-L6-v2", {
          progress_callback: (data) => {
            if(data.status==='progress'){
              self.postMessage({type: 'download', data})
          }     
        }});

        const getEmbed = async (text) => {
          let out = await pipe1(text);
          return out.data
        }

        const sortArtEmbeds = async (articles, queryEmbed) => {
          const sents = getSents(articles)
          const embeds = await Promise.all(sents.map(async e => [e.segment, calculateCosineSimilarity(queryEmbed, await getEmbed(e.segment))]))
          embeds.sort((a, b) => b[1] - a[1])
          return embeds
        }

        const queryEmbed = await getEmbed(prompt.terms)
        const embeds = await sortArtEmbeds(prompt.articles, queryEmbed)
        const generation = embeds.slice(0,3).map(e => e[0]) // top 3 scores (most similiar)
      
        return self.postMessage({type: task, generation, then});
      } catch(error){
        self.postMessage({type: 'error', error});
      }
    }

    try {
      let pipe = await pipeline('text2text-generation', model, {
          progress_callback: (data) => {
            if(data.status==='progress'){
              self.postMessage({type: 'download', data})
          }     
      }});

      let out = await pipe(prompt);
      const generation = out[0]
      return self.postMessage({type: task, generation, then});
      
    } catch(error){
      self.postMessage({type: 'error', error});
    }
});