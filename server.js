import express from 'express';
import axios from 'axios';
import dotenv from 'dotenv';
import cors from 'cors';
import { extract } from '@extractus/article-extractor';

dotenv.config();

const app = express();
const port = 5000;

app.use(cors());
app.use(express.json());

// Utility function to split text into smaller chunks
function splitTextIntoChunks(text, chunkSize = 1500) {
    let chunks = [];
    for (let i = 0; i < text.length; i += chunkSize) {
        chunks.push(text.slice(i, i + chunkSize));
    }
    return chunks;
}

// Summarization endpoint
app.post('/summarize', async (req, res) => {
    let { text } = req.body;

    try {
        // If text looks like a URL, extract article content
        if (text.startsWith('http://') || text.startsWith('https://')) {
            const article = await extract(text);
            if (!article || !article.content) {
                return res.status(400).json({ error: 'Failed to extract content from URL.' });
            }
            text = article.content;
        }

        // Step 1: Split long text into chunks
        const chunks = splitTextIntoChunks(text);

        // Step 2: Summarize each chunk
        const chunkSummaries = await Promise.all(
            chunks.map(async (chunk) => {
                const response = await axios.post(
                    'https://api-inference.huggingface.co/models/facebook/bart-large-cnn',
                    {
                        inputs: chunk,
                        parameters: {
                            max_length: 150,
                            min_length: 100,
                            do_sample: true,
                            num_beams: 4,
                            temperature: 0.7,
                            top_k: 50,
                            top_p: 0.95,
                            repetition_penalty: 1.2,
                            no_repeat_ngram_size: 3,
                            length_penalty: 1.0,
                            early_stopping: true
                        }
                    },
                    {
                        headers: {
                            Authorization: `Bearer ${process.env.HUGGINGFACE_API_KEY}`,
                            'Content-Type': 'application/json'
                        }
                    }
                );
                return response.data[0].summary_text;
            })
        );

        // Step 3: Combine the chunk summaries into a single summary
        const combinedSummary = chunkSummaries.join(' ');

        // Final summarization of the combined summary
        const finalResponse = await axios.post(
            'https://api-inference.huggingface.co/models/facebook/bart-large-cnn',
            {
                inputs: combinedSummary,
                parameters: {
                    max_length: 150,
                    min_length: 100,
                    do_sample: true,
                    num_beams: 4,
                    temperature: 0.7,
                    top_k: 50,
                    top_p: 0.95,
                    repetition_penalty: 1.2,
                    no_repeat_ngram_size: 3,
                    length_penalty: 1.0,
                    early_stopping: true
                }
            },
            {
                headers: {
                    Authorization: `Bearer ${process.env.HUGGINGFACE_API_KEY}`,
                    'Content-Type': 'application/json'
                }
            }
        );

        const summary = finalResponse.data[0].summary_text;

        // Send final summary
        res.json({ summary });

    } catch (error) {
        console.error('Error summarizing:', error.message);
        if (error.response) {
            console.error('Hugging Face error details:', error.response.data);
        }
        res.status(500).json({ error: 'Summarization failed' });
    }
});

app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});
