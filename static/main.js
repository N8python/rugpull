import * as THREE from './three.module.min.js';
import { FullScreenQuad } from './Pass.js';
import { OrbitControls } from './OrbitControls.js';
import * as pdfjsLib from './pdf.mjs'
pdfjsLib.GlobalWorkerOptions.workerSrc = './pdf.worker.mjs';


function splitIntoSentences(text) {
    const alphabets = "([A-Za-z])";
    const prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]";
    const suffixes = "(Inc|Ltd|Jr|Sr|Co)";
    const starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\\s|She\\s|It\\s|They\\s|Their\\s|Our\\s|We\\s|But\\s|However\\s|That\\s|This\\s|Wherever)";
    const acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)";
    const websites = "[.](com|net|org|io|gov|me|edu)";
    const digits = "([0-9])";
    const multipleDots = /(\.{2,})/g;

    text = " " + text + "  ";
    text = text.replace(/\n/g, " ");
    text = text.replace(new RegExp(prefixes, "g"), "\\1<prd>");
    text = text.replace(new RegExp(websites, "g"), "<prd>$1");
    text = text.replace(new RegExp(digits + "[.]" + digits, "g"), "$1<prd>$2");
    text = text.replace(multipleDots, (match) => "<prd>".repeat(match.length) + "<stop>");
    if (text.includes("Ph.D")) text = text.replace("Ph.D.", "Ph<prd>D<prd>");
    text = text.replace(new RegExp("\\s" + alphabets + "[.] ", "g"), " $1<prd> ");
    text = text.replace(new RegExp(acronyms + " " + starters, "g"), "$1<stop> $2");
    text = text.replace(new RegExp(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "g"), "$1<prd>$2<prd>$3<prd>");
    text = text.replace(new RegExp(alphabets + "[.]" + alphabets + "[.]", "g"), "$1<prd>$2<prd>");
    text = text.replace(new RegExp(" " + suffixes + "[.] " + starters, "g"), " $1<stop> $2");
    text = text.replace(new RegExp(" " + suffixes + "[.]", "g"), " $1<prd>");
    text = text.replace(new RegExp(" " + alphabets + "[.]", "g"), " $1<prd>");
    text = text.replace(new RegExp("\\.\\.\\.", "g"), "<prd><prd><prd>");
    if (text.includes("”")) text = text.replace(/.”/g, "”.");
    if (text.includes("\"")) text = text.replace(/."/, "\".");
    if (text.includes("!")) text = text.replace(/!"/, "\"!");
    if (text.includes("?")) text = text.replace(/\?"/, "\"?");
    text = text.replace(/\./g, ".<stop>");
    text = text.replace(/\?/g, "?<stop>");
    text = text.replace(/!/g, "!<stop>");
    text = text.replace(/<prd>/g, ".");
    let sentences = text.split("<stop>");
    sentences = sentences.map(s => s.trim());
    if (sentences.length > 0 && sentences[sentences.length - 1] === "") {
        sentences.pop();
    }
    return sentences;
}

function updateProgressBar(value) {
    const progressBar = document.querySelector('.progress');
    const maxWidth = document.querySelector('.progress-container').offsetWidth;
    const width = Math.max(0, Math.min(maxWidth, value * (maxWidth)));
    progressBar.style.width = `${width}px`;
}

function cosineSimilarity(a, b, mag2) {
    const l = a.length;
    let dot = 0;
    let mag1 = 0;
    for (let i = 0; i < l; i++) {
        dot += a[i] * b[i];
        mag1 += a[i] * a[i];
    }
    return dot / (Math.sqrt(mag1) * mag2);
}

function cosineSimilarity2(a, b) {
    const l = a.length;
    let dot = 0;
    let mag1 = 0;
    let mag2 = 0;
    for (let i = 0; i < l; i++) {
        dot += a[i] * b[i];
        mag1 += a[i] * a[i];
        mag2 += b[i] * b[i];
    }
    return dot / (Math.sqrt(mag1) * Math.sqrt(mag2));
}

function simpleTopK(arr, k) {
    const maxIndices = {};

    for (let i = 0; i < k; i++) {
        let maxIndex = 0;
        let maxValue = -Infinity;
        for (let j = 0; j < arr.length; j++) {
            if (arr[j] > maxValue && !(j in maxIndices)) {
                maxIndex = j;
                maxValue = arr[j];
            }
        }
        maxIndices[maxIndex] = maxValue;
    }
    return maxIndices;
}

function cosine(x, y) {
    let result = 0.0;
    let normX = 0.0;
    let normY = 0.0;

    for (let i = 0; i < x.length; i++) {
        result += x[i] * y[i];
        normX += x[i] * x[i];
        normY += y[i] * y[i];
    }

    if (normX === 0 && normY === 0) {
        return 0;
    } else if (normX === 0 || normY === 0) {
        return 1.0;
    } else {
        return 1.0 - result / Math.sqrt(normX * normY);
    }
}

function computeContextAndSentenceRanges(sentences, WINDOW_SIZE, WINDOW_STRIDE) {
    const contexts = [];
    const sentenceRanges = [];
    for (let i = 0; i < sentences.length - WINDOW_SIZE; i += WINDOW_STRIDE) {
        const context = sentences.slice(i, i + WINDOW_SIZE);
        contexts.push(context.join(" "));
        sentenceRanges.push([i, i + WINDOW_SIZE])
    }
    return { contexts, sentenceRanges };
}
let embedding_f32, contexts, sentenceRanges, results, embeddingsTexSize;
let timelineArr, timelineArrUnnormalized;
let sentences, WINDOW_SIZE = 1,
    WINDOW_STRIDE = 5,
    batchSize, totalBatches, EMBEDDING_SIZE = 384,
    embeddingsArray, umap, embedding_u8, contextMinMax;
let topKcontexts = [];
const geometry = new THREE.BufferGeometry();

let query = new Float32Array(EMBEDDING_SIZE);
const embeddingsQuad = new FullScreenQuad(new THREE.ShaderMaterial({
    uniforms: {
        //   embeddings: { value: embeddingsDataTexture },
        embeddingTextureSize: { value: null },
        embeddings_u32: { value: null },
        minmaxQuantTexture: { value: null },
    },
    vertexShader: /*glsl*/ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4( position, 1.0 );
}
`,
    fragmentShader: /*glsl*/ `
precision highp sampler3D;
precision highp usampler3D;
uniform sampler3D embeddings;
uniform usampler3D embeddings_u32;
uniform sampler2D minmaxQuantTexture;
uniform float embeddingTextureSize;
uniform EmbeddingData {
    vec4 inputQuery[${EMBEDDING_SIZE / 4}];
};
varying vec2 vUv;
vec4 unpack4x8(uint packed, float minembedding, float maxembedding) {
    float a = float((packed >> 24) & 0xFFu) / 255.0;
    float b = float((packed >> 16) & 0xFFu) / 255.0;
    float c = float((packed >> 8) & 0xFFu) / 255.0;
    float d = float((packed >> 0) & 0xFFu) / 255.0;
    vec4 result = vec4(a, b, c, d);
    result = result * (maxembedding - minembedding) + minembedding;
    return result;
}
void main() {
    vec2 pixelCoord = floor(vUv * vec2(embeddingTextureSize, embeddingTextureSize));
    float sum = 0.0;
    float mag1 = 0.0;
    vec2 quantData = texelFetch(minmaxQuantTexture, ivec2(pixelCoord), 0).rg;
    float minembedding = quantData.r;
    float maxembedding = quantData.g;

    for (int i = 0; i < ${EMBEDDING_SIZE / 16}; i++) {
        uvec4 packed4 = texelFetch(embeddings_u32, ivec3(pixelCoord, i), 0);
        vec4 a = unpack4x8(packed4.x, minembedding, maxembedding);
        vec4 b = inputQuery[i * 4];
        sum += dot(a, b);

        vec4 a2 = unpack4x8(packed4.y, minembedding, maxembedding);
        vec4 b2 = inputQuery[i * 4 + 1];
        sum += dot(a2, b2);

        vec4 a3 = unpack4x8(packed4.z, minembedding, maxembedding);
        vec4 b3 = inputQuery[i * 4 + 2];
        sum += dot(a3, b3);

        vec4 a4 = unpack4x8(packed4.w, minembedding, maxembedding);
        vec4 b4 = inputQuery[i * 4 + 3];
        sum += dot(a4, b4);
    }
    float similarity = sum;
    gl_FragColor = vec4(similarity, similarity, similarity, 1.0);
}
`
}));
const embeddingsUniformGroup = new THREE.UniformsGroup();
const embeddingUniformArray = [];
for (let i = 0; i < EMBEDDING_SIZE / 4; i++) {
    embeddingUniformArray.push(new THREE.Uniform(new THREE.Vector4()));
}
embeddingsUniformGroup.setName("EmbeddingData");
embeddingsUniformGroup.add(embeddingUniformArray);

embeddingsQuad.material.uniformsGroups = [embeddingsUniformGroup];
const embeddingsRenderTarget = new THREE.WebGLRenderTarget(
    1, 1, {
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
        format: THREE.RedFormat,
        type: THREE.FloatType
    }
);
const rendererGraph = new THREE.WebGLRenderer({ antialias: true, premultipliedAlpha: false });
rendererGraph.setPixelRatio(2)
rendererGraph.setSize(512, 512);
rendererGraph.domElement.style.width = "512px";
rendererGraph.domElement.style.height = "512px";
rendererGraph.domElement.style.borderRadius = "4px";
const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1000);
camera.position.set(0, 0, 5);
const material =
    new THREE.ShaderMaterial({
        uniforms: {
            selectedId: { value: -1 },
            zoom: { value: camera.zoom },
            size: { value: 40.0 }
        },
        vertexShader: `
            attribute float intensity;
            attribute float id;
            uniform float selectedId;
            uniform float size;
            uniform float zoom;
            varying float vIntensity;
            varying float vId;
            void main() {
                vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );
                mvPosition.z += (selectedId == id ? 1.0 : intensity) * 0.1;
                gl_Position = projectionMatrix * mvPosition;
                gl_PointSize = size * zoom;
                vIntensity = intensity;
                vId = id;
                }
            `,
        fragmentShader: `
            varying float vIntensity;
            varying float vId;
            uniform float selectedId;
            float saturate( float x ) { return clamp( x, 0.0, 1.0 ); }

vec3 viridis_quintic( float x )
{
x = saturate( x );
vec4 x1 = vec4( 1.0, x, x * x, x * x * x ); // 1 x x2 x3
vec4 x2 = x1 * x1.w * x; // x4 x5 x6 x7
return vec3(
dot( x1.xyzw, vec4( +0.280268003, -0.143510503, +2.225793877, -14.815088879 ) ) + dot( x2.xy, vec2( +25.212752309, -11.772589584 ) ),
dot( x1.xyzw, vec4( -0.002117546, +1.617109353, -1.909305070, +2.701152864 ) ) + dot( x2.xy, vec2( -1.685288385, +0.178738871 ) ),
dot( x1.xyzw, vec4( +0.300805501, +2.614650302, -12.019139090, +28.933559110 ) ) + dot( x2.xy, vec2( -33.491294770, +13.762053843 ) ) );
}

            void main() {
                float l = length(gl_PointCoord.xy * 2.0 - 1.0);
                if (l > 1.0) discard;
                gl_FragColor = vec4(viridis_quintic(pow(vIntensity, 2.5)), 1.0);
                if (vId == selectedId && l > 0.8) {
                    gl_FragColor = vec4(vec3(1.0), 1.0);
                }
            }
            `
    });

const idRenderTarget = new THREE.WebGLRenderTarget(
    1024, 1024, {
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
        format: THREE.RedFormat,
        type: THREE.FloatType
    }
);

document.getElementById("graphCanvas").appendChild(rendererGraph.domElement);
document.getElementById("saveFileButton").addEventListener('click', () => {
    const savefilePath = document.getElementById("savefileTxt").value;
    const url = 'http://localhost:3000/save';
    fetch(url, {
        method: 'POST', // or 'PUT'
        headers: {
            'Content-Type': 'application/json',
            noCors: true
        },

        body: JSON.stringify({
            filename: savefilePath,
            data: {
                sentences,
                WINDOW_SIZE,
                WINDOW_STRIDE,
            }
        }), // Convert JavaScript object to a JSON string
    });
    // Now use binary data to save the rest
    let binaryData = new Blob([embedding_u8.buffer], { type: "application/octet-stream" });
    let filePath = savefilePath;

    let formData = new FormData();
    formData.append('file', binaryData);
    formData.append('path', filePath);

    fetch('http://localhost:3000/save-embeddings-u8', {
        method: 'POST',
        body: formData
    });

    binaryData = new Blob([contextMinMax.buffer], { type: "application/octet-stream" });
    filePath = savefilePath;

    formData = new FormData();
    formData.append('file', binaryData);
    formData.append('path', filePath);

    fetch('http://localhost:3000/save-embeddings-ctx', { method: 'POST', body: formData });

    /* const resultBinaryData = new Float32Array(results.length * 3);
    for (let i = 0; i < results.length; i++) {
        resultBinaryData[i * 3] = results[i][0];
        resultBinaryData[i * 3 + 1] = results[i][1];
        resultBinaryData[i * 3 + 2] = results[i][2];
    }
*/
    binaryData = new Blob([results.buffer], { type: "application/octet-stream" });
    filePath = savefilePath;

    formData = new FormData();
    formData.append('file', binaryData);
    formData.append('path', filePath);

    fetch('http://localhost:3000/save-results', { method: 'POST', body: formData });




    filePath = savefilePath;
});

function submitVector(vec) {
    for (let i = 0; i < EMBEDDING_SIZE / 4; i++) {
        embeddingUniformArray[i].value.set(
            vec[i * 4 + 0],
            vec[i * 4 + 1],
            vec[i * 4 + 2],
            vec[i * 4 + 3]
        );
    }
}
document.getElementById("search-input").addEventListener('input', async() => {
    const start = performance.now();
    const searchTextInput = document.getElementById("search-input");
    let searchQuery = searchTextInput.value;
    let negativeQuery = "";
    if (searchQuery.includes("--no")) {
        negativeQuery = searchQuery.split("--no")[1].trim();
        searchQuery = searchQuery.split("--no")[0].trim();
    }
    const url = 'http://localhost:3000/post';
    const result = await fetch(url, {
        method: 'POST', // or 'PUT'
        headers: {
            'Content-Type': 'application/json',
            noCors: true
        },

        body: JSON.stringify([searchQuery, negativeQuery]), // Convert JavaScript object to a JSON string
    });
    const buf = await result.arrayBuffer();
    const embedding_query = new Float32Array(buf.slice(0, EMBEDDING_SIZE * 4));
    const embedding_query_neg = new Float32Array(buf.slice(EMBEDDING_SIZE * 4, EMBEDDING_SIZE * 8));
    query.set(embedding_query);
    const similarityScores = new Float32Array(contexts.length);
    console.time("GPU Similarity Score Search");
    rendererGraph.setRenderTarget(embeddingsRenderTarget);
    submitVector(embedding_query);
    embeddingsQuad.render(rendererGraph);
    const r = new Float32Array(embeddingsTexSize * embeddingsTexSize);
    rendererGraph.readRenderTargetPixels(embeddingsRenderTarget, 0, 0, embeddingsTexSize, embeddingsTexSize, r);
    const similarityScoresGPU = r.slice(0, contexts.length);
    similarityScores.set(similarityScoresGPU);
    rendererGraph.setRenderTarget(null);

    if (negativeQuery !== "") {
        query.set(embedding_query_neg);
        rendererGraph.setRenderTarget(embeddingsRenderTarget);
        submitVector(embedding_query_neg);
        embeddingsQuad.render(rendererGraph);
        const r = new Float32Array(embeddingsTexSize * embeddingsTexSize);
        rendererGraph.readRenderTargetPixels(embeddingsRenderTarget, 0, 0, embeddingsTexSize, embeddingsTexSize, r);
        const similarityScoresGPUNeg = r.slice(0, contexts.length);
        for (let i = 0; i < similarityScores.length; i++) {
            similarityScores[i] -= similarityScoresGPUNeg[i];
        }
        rendererGraph.setRenderTarget(null);
    }

    console.timeEnd("GPU Similarity Score Search");
    const topK = Object.entries(simpleTopK(similarityScores, 5)).sort((a, b) => b[1] - a[1]);
    // console.log(topK);
    const results = document.getElementById("results");
    results.innerHTML = "";
    topKcontexts = [];
    let topKCtr = 0;
    for (const [index, score] of topK) {
        topKCtr++;
        const result = document.createElement("div");
        // result.innerText = `${score.toFixed(2)}: ${contexts[index]}`;
        result.innerHTML = `
    <div class="result" id="context${topKCtr}">
      <h3 class="result-title">#${topKCtr}. Sentences ${sentenceRanges[index][0]}-${sentenceRanges[index][1]}</h3>
      <h3 class="result-score">Similarity: ${score.toFixed(2)}</h3>
     <div class="result-text">${contexts[index]}</div>
    </div>
   `;
        results.appendChild(result);
        topKcontexts.push({
            index,
            score
        })
    }
    timelineArr.set(similarityScores);
    timelineArrUnnormalized.set(similarityScores);
    // Normalize timelineArr
    let maxScore = -Infinity;
    let minScore = Infinity;
    for (let i = 0; i < timelineArr.length; i++) {
        maxScore = Math.max(maxScore, timelineArr[i]);
        minScore = Math.min(minScore, timelineArr[i]);
    }
    for (let i = 0; i < timelineArr.length; i++) {
        timelineArr[i] = (timelineArr[i] - minScore) / (maxScore - minScore);
        timelineArr[i] = timelineArr[i];
    }
    geometry.attributes.intensity.needsUpdate = true;
    document.getElementById("graph").style.display = "flex";
    const end = performance.now();
    console.log(`Similarity calculation took ${end - start}ms`);
});
const idMaterial = new THREE.ShaderMaterial({
    uniforms: {
        selectedId: { value: -1 },
        zoom: { value: camera.zoom },
        size: { value: 40.0 }
    },
    vertexShader: `
attribute float intensity;
attribute float id;
uniform float size;
uniform float zoom;
varying float vIntensity;
varying float vId;
void main() {
    vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );
    mvPosition.z += intensity * 0.1;
    gl_Position = projectionMatrix * mvPosition;
    gl_PointSize = size * zoom;
    vIntensity = intensity;
    vId = id;
    }
`,
    fragmentShader: `
varying float vId;
void main() {
    float l = length(gl_PointCoord.xy * 2.0 - 1.0);
    if (l > 1.0) discard;
    gl_FragColor = vec4(vId, 0.0, 0.0, 1.0);
}
`
});
const points = new THREE.Points(geometry, material);
points.frustumCulled = false;
scene.add(points);
const queryEmbed = new THREE.Mesh(
    new THREE.CircleGeometry(1.0, 32, 32),
    new THREE.ShaderMaterial({
        transparent: true,
        vertexShader: `
    varying vec3 vPosition;
    void main() {
        vPosition = position;
        gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
    }
    `,
        fragmentShader: `
    varying vec3 vPosition;
    void main() {
        float l = length(vPosition.xy);
        gl_FragColor = vec4(1.0, 0.0, 0.0, max(1.0 - l, 0.0));
    }
    `
    })
);
queryEmbed.scale.set(0.2, 0.2, 1.0);
queryEmbed.position.set(0, 0, 0.11);
const controls = new OrbitControls(camera, rendererGraph.domElement);
controls.target.set(0, 0, 0.2);
controls.enableRotate = false;

function getMousePos(canvas, evt) {
    var rect = canvas.getBoundingClientRect();
    return {
        x: evt.clientX - rect.left,
        y: evt.clientY - rect.top
    };
}
const canvas = rendererGraph.domElement;
let mouseDown = false;

function updateId(evt) {
    var mousePos = getMousePos(canvas, evt);
    if (mousePos.x < 0 || mousePos.x > 512 || mousePos.y < 0 || mousePos.y > 512) {
        return;
    }
    rendererGraph.setRenderTarget(idRenderTarget);
    rendererGraph.setClearColor(new THREE.Color(-1, -1, -1));
    points.material = idMaterial;
    rendererGraph.render(points, camera);
    rendererGraph.setClearColor(new THREE.Color(0, 0, 0));
    points.material = material;
    camera.clearViewOffset();
    rendererGraph.setRenderTarget(null);

    const pixelBuffer = new Float32Array(1);
    rendererGraph.readRenderTargetPixels(idRenderTarget, Math.floor(mousePos.x * 2), 1024 - Math.floor(mousePos.y * 2), 1, 1, pixelBuffer);
    const id = pixelBuffer[0];
    material.uniforms.selectedId.value = id;
    if (id > -1) {
        const result = document.getElementById("result");
        const score = timelineArrUnnormalized[id];
        result.innerHTML = `
    <div class="result" style="height:100%;flex-grow:1;">
      <h3 class="result-title">Sentences ${sentenceRanges[id][0]}-${sentenceRanges[id][1]}</h3>
      <h3 class="result-score">Similarity: ${score.toFixed(2)}</h3>
     <div class="result-text">${contexts[id]}</div>
    </div>
   `;

    } else {
        const result = document.getElementById("result");
        result.innerHTML = `
    <div class="result" style="height:100%;flex-grow:1;">
      <h3 class="result-title">Click on a point to see the sentence</h3>
        <div class="result-text">No sentence selected.</div>

    </div>
   `;
    }
}
document.addEventListener('mousedown', function(evt) {
    if (evt.button !== 0) return;
    mouseDown = true;
    updateId(evt);
});
document.addEventListener('mouseup', function(evt) {
    mouseDown = false;
});
canvas.addEventListener('mousemove', function(evt) {
    if (!mouseDown) {
        return;
    }
    updateId(evt);
});
document.getElementById("search-button").addEventListener('click', async() => {
    const answerArea = document.getElementById("answer-area");
    answerArea.style.display = "block";
    answerArea.innerText = "Answering...";
    const systemPrompt = "You are an intelligent AI that takes in a query and relevant contexts. Given the context and the query, answer the question. Contexts won't always have the info you need - you should also rely on your pre-existing knowledge base. Cite contexts parenthetically - for context 2, put (C:2) at the end of the sentence. Cite your external knowledge base (C:OI), for outside information.";
    const searchTextInput = document.getElementById("search-input");
    const searchQuery = searchTextInput.value.split("--no")[0].trim();
    const relevantContexts = topKcontexts.map((x, i) => `${i + 1} (Sentences ${sentenceRanges[x.index][0]}-${sentenceRanges[x.index][1]}). ${contexts[x.index]}`).join("\n");
    const userQuery = `Query: ${searchQuery}

Relevant Contexts:
${relevantContexts}
                    
Answer (MAKE SURE to CITE SOURCES with the string '(C:N)' where N is the context number as a PARENTHETICAL CITATION after the RELEVANT INFORMATION. If there are NO RELEVANT SOURCES cite OUTSIDE INFORMATION with '(C:OI)' - use this ONLY IF NECESSARY and if you are VERY CONFIDENT in YOUR KNOWLEDGE. Cite MULTIPLE relevant sources with '(C:N,C:M)'. Cite a VARIETY of sources in your answer. Answer the question DIRECTLY and BE SPECIFIC.):`
    let accumulator = "";
    fetch('http://localhost:1234/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                messages: [
                    { role: 'system', content: systemPrompt },
                    {
                        role: 'user',
                        content: userQuery
                    }
                ],
                temperature: 0.7,
                max_tokens: -1,
                stream: true
            })
        }).then(response => {
            const reader = response.body.getReader();

            // Function to recursively read each chunk
            function read() {
                reader.read().then(({ done, value }) => {
                    if (done) {
                        console.log('Stream finished.');
                        return;
                    }
                    // Log each chunk as it comes
                    const raw = new TextDecoder().decode(value);
                    // Extract the JSON object {.+}
                    const match = raw.match(/\{.+\}/);
                    if (match) {
                        const c = JSON.parse(match[0]);

                        const answer = c.choices[0].delta.content;
                        if (c.choices[0].delta.role === "assistant") {
                            accumulator += answer;
                            let toDisplay = accumulator;
                            // Replace each instance of (C:X) with <a href="#contextx">x</a>
                            toDisplay = toDisplay.replace(/C:(\d)/g, '<a href="#context$1">$1</a>');
                            toDisplay = toDisplay.replace(/C:OI/g, '<a href="https://huggingface.co/mlabonne/NeuralBeagle14-7B">OI</a>');
                            answerArea.innerHTML = toDisplay;
                        }


                    }
                    // Read the next chunk
                    read();
                });
            }

            // Start reading the stream
            read();
        })
        /*console.log(c)
                /*const answer = c.choices[0].message.content;
                // Replace each instance of context X with <a href="#contextx">x</a>
                answerArea.innerHTML = answer.replace(/C:(\d)/g, '<a href="#context$1">$1</a>');*/


});
let doUpdate = false;

function animate() {
    if (!doUpdate) {
        requestAnimationFrame(animate);
        return;
    }
    updateProgressBar(1);
    controls.update();
    requestAnimationFrame(animate);
    material.uniforms.zoom.value = camera.zoom;
    idMaterial.uniforms.zoom.value = camera.zoom;
    const slideScale = document.getElementById("pointSize").value;
    material.uniforms.size.value = 40.0 * slideScale;
    idMaterial.uniforms.size.value = 40.0 * slideScale;
    rendererGraph.render(scene, camera);
}

requestAnimationFrame(animate);

function clear() {
    document.getElementById("search-input").value = "";
    document.getElementById("graph").style.display = "none";
    document.getElementById("result").innerHTML = `
    <div class="result" style="height:100%;flex-grow:1;">
    <h3 class="result-title">Click over a point to see the sentence</h3>
    <div class="result-text">No sentence selected.</div>
    </div>
    `;
    document.getElementById("results").innerHTML = "";
    document.getElementById("answer-area").style.display = "none";
}
async function loadFileOptions() {
    const options = await (await fetch('http://localhost:3000/list-files', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ path: "cache" })

    })).json();
    const select = document.getElementById("loadfileTxt");
    select.innerHTML = "";
    for (const option of options) {
        const opt = document.createElement('option');
        opt.value = option;
        opt.innerHTML = option;
        select.appendChild(opt);
    }
    document.getElementById("loadFileButton").disabled = false;

}
loadFileOptions();
async function main(options) {
    doUpdate = false;
    clear();

    function computeQuantInfo(embedding_f32) {
        const embeddingsTexSize = Math.ceil(Math.sqrt(contexts.length));
        const contextMinMax = new Float32Array(embeddingsTexSize * embeddingsTexSize * 2);
        for (let i = 0; i < contexts.length; i++) {
            const contextArr = embedding_f32.slice(i * EMBEDDING_SIZE, (i + 1) * EMBEDDING_SIZE);
            let minembedding = Infinity;
            let maxembedding = -Infinity;
            for (let j = 0; j < contextArr.length; j++) {
                minembedding = contextArr[j] < minembedding ? contextArr[j] : minembedding;
                maxembedding = contextArr[j] > maxembedding ? contextArr[j] : maxembedding;
            }
            const xCoord = i % embeddingsTexSize;
            const yCoord = Math.floor(i / embeddingsTexSize);
            contextMinMax[yCoord * embeddingsTexSize * 2 + xCoord * 2] = minembedding;
            contextMinMax[yCoord * embeddingsTexSize * 2 + xCoord * 2 + 1] = maxembedding;
        }


        const embedding_u8 = new Uint8Array(embedding_f32.length);
        for (let i = 0; i < contexts.length; i++) {
            const contextArr = embedding_f32.slice(i * EMBEDDING_SIZE, (i + 1) * EMBEDDING_SIZE);
            const xCoord = i % embeddingsTexSize;
            const yCoord = Math.floor(i / embeddingsTexSize);
            const minembedding = contextMinMax[yCoord * embeddingsTexSize * 2 + xCoord * 2];
            const maxembedding = contextMinMax[yCoord * embeddingsTexSize * 2 + xCoord * 2 + 1];
            for (let j = 0; j < contextArr.length; j++) {
                embedding_u8[i * EMBEDDING_SIZE + j] = Math.min(Math.round((contextArr[j] - minembedding) / (maxembedding - minembedding) * 255), 255);
            }
        }
        return {
            embedding_u8: embedding_u8,
            contextMinMax: contextMinMax
        }
    }

    if (options.loadFile) {
        // embedding_f32 = Float32Array.from(options.data.embeddings);
        sentences = options.data.sentences;
        embedding_u8 = Uint8Array.from(options.data.embedding_u8);
        contextMinMax = Float32Array.from(options.data.contextMinMax);
        contexts = options.data.contexts;
        sentenceRanges = options.data.sentenceRanges;
        results = options.data.results;

        WINDOW_SIZE = 5;
        WINDOW_STRIDE = 1;
        batchSize = 128;
        totalBatches = Math.ceil(contexts.length / batchSize);
        EMBEDDING_SIZE = 384;

    } else {
        console.time();
        sentences = splitIntoSentences(options.textFile);
        WINDOW_SIZE = 5;
        WINDOW_STRIDE = 1;
        contexts = [];
        sentenceRanges = [];
        const contextAndSentenceRanges = computeContextAndSentenceRanges(sentences, WINDOW_SIZE, WINDOW_STRIDE);
        contexts = contextAndSentenceRanges.contexts;
        sentenceRanges = contextAndSentenceRanges.sentenceRanges;
        console.timeEnd();
        batchSize = 128;
        console.time('Full Embeddings');
        totalBatches = Math.ceil(contexts.length / batchSize);
        EMBEDDING_SIZE = 384;
        embedding_f32 = new Float32Array(contexts.length * EMBEDDING_SIZE);
        document.getElementById("progress-label").innerText = `0/${contexts.length} (0%) chunks embedded in 0ms`;
        let runningTime = 0;
        for (let i = 0; i < contexts.length; i += batchSize) {
            const start = performance.now();
            const timerKey = `Batch ${i / batchSize + 1}/${totalBatches}`;
            console.time(timerKey);
            const batch = contexts.slice(i, i + batchSize);
            const url = 'http://localhost:3000/post';
            const result = await fetch(url, {
                method: 'POST', // or 'PUT'
                headers: {
                    'Content-Type': 'application/json',
                    noCors: true
                },

                body: JSON.stringify(batch), // Convert JavaScript object to a JSON string
            })
            const buf = await result.arrayBuffer();
            const buf_f32 = new Float32Array(buf);
            embedding_f32.set(buf_f32, i * EMBEDDING_SIZE);
            console.timeEnd(timerKey);
            runningTime += performance.now() - start;
            updateProgressBar(Math.min((i + batchSize) / contexts.length, 1));
            const timeRemaining = runningTime / (i + batchSize) * (contexts.length - (i + batchSize));
            document.getElementById("progress-label").innerText = `${Math.min(i + batchSize, contexts.length)}/${contexts.length} (${((Math.min(i + batchSize, contexts.length) / contexts.length) * 100).toFixed(2)}%) chunks embedded in ${(runningTime / 1000).toFixed(1)}s, ${(timeRemaining / 1000).toFixed(1)}s remaining`;
        }
        console.timeEnd('Full Embeddings');
        embeddingsArray = [];
        for (let i = 0; i < contexts.length; i++) {
            embeddingsArray.push(embedding_f32.slice(i * EMBEDDING_SIZE, (i + 1) * EMBEDDING_SIZE));
        }
        console.time();
        document.getElementById("progress-label").innerText = `Initializing UMAP...`;
        updateProgressBar(0);
        await new Promise(resolve => setTimeout(resolve, 100));
        umap = new UMAP({
            distanceFn: cosine,
        });

        results = new Float32Array(await (await fetch('http://localhost:3000/post-umap', {
            method: 'POST', // or 'PUT'
            headers: {
                'Content-Type': 'application/octet-stream',
                noCors: true
            },
            body: embedding_f32.buffer
        })).arrayBuffer());
        const oldResults = results;

        results = new Float32Array(contexts.length * 3);
        for (let i = 0; i < oldResults.length; i++) {
            results[i * 3] = oldResults[i * 2];
            results[i * 3 + 1] = oldResults[i * 2 + 1];
            results[i * 3 + 2] = 0;
        }
        console.timeEnd();
        document.getElementById("progress-label").innerText = `Done!`;
        //{ embedding_u8, contextMinMax } = computeQuantInfo(embedding_f32);
        const quantInfo = computeQuantInfo(embedding_f32);
        embedding_u8 = quantInfo.embedding_u8;
        contextMinMax = quantInfo.contextMinMax;
    }




    function setitup(embedding_u8, contextMinMax, contexts, sentenceRanges, results) {
        embeddingsTexSize = Math.ceil(Math.sqrt(contexts.length));
        const minmaxQuantTexture = new THREE.DataTexture(contextMinMax, embeddingsTexSize, embeddingsTexSize, THREE.RGFormat, THREE.FloatType);
        minmaxQuantTexture.needsUpdate = true;
        const embeddingsDataArrayU32 = new Uint32Array(embeddingsTexSize * embeddingsTexSize * EMBEDDING_SIZE / 4);
        // Crazy packing time - we pack four quantized u8s into a single u32, then pack those u32s into a singular texel, letting us use a single texel fetch to get all *sixteen* values
        function pack4u8tou32(a, b, c, d) {
            return (a << 24) | (b << 16) | (c << 8) | d;
        }
        for (let i = 0; i < contexts.length; i++) {
            const yCoord = Math.floor(i / embeddingsTexSize);
            const xCoord = i % embeddingsTexSize;
            for (let j = 0; j < EMBEDDING_SIZE / 16; j++) {
                // Packing time
                const startIdx = i * EMBEDDING_SIZE + j * 16;
                const packed0 = pack4u8tou32(embedding_u8[startIdx], embedding_u8[startIdx + 1], embedding_u8[startIdx + 2], embedding_u8[startIdx + 3]);
                const packed1 = pack4u8tou32(embedding_u8[startIdx + 4], embedding_u8[startIdx + 5], embedding_u8[startIdx + 6], embedding_u8[startIdx + 7]);
                const packed2 = pack4u8tou32(embedding_u8[startIdx + 8], embedding_u8[startIdx + 9], embedding_u8[startIdx + 10], embedding_u8[startIdx + 11]);
                const packed3 = pack4u8tou32(embedding_u8[startIdx + 12], embedding_u8[startIdx + 13], embedding_u8[startIdx + 14], embedding_u8[startIdx + 15]);

                const zCoord = j;
                const idx = 4 * (zCoord * embeddingsTexSize * embeddingsTexSize + yCoord * embeddingsTexSize + xCoord);
                embeddingsDataArrayU32[idx] = packed0;
                embeddingsDataArrayU32[idx + 1] = packed1;
                embeddingsDataArrayU32[idx + 2] = packed2;
                embeddingsDataArrayU32[idx + 3] = packed3;

            }
        }
        const embeddingsDataTextureU32 = new THREE.Data3DTexture(embeddingsDataArrayU32, embeddingsTexSize, embeddingsTexSize, EMBEDDING_SIZE / 16);
        embeddingsDataTextureU32.format = THREE.RGBAIntegerFormat;
        embeddingsDataTextureU32.type = THREE.UnsignedIntType;
        embeddingsDataTextureU32.internalFormat = 'RGBA32UI';
        embeddingsDataTextureU32.needsUpdate = true;

        embeddingsRenderTarget.setSize(embeddingsTexSize, embeddingsTexSize);

        embeddingsQuad.material.uniforms.minmaxQuantTexture.value = minmaxQuantTexture;
        embeddingsQuad.material.uniforms.embeddingTextureSize.value = embeddingsTexSize;
        embeddingsQuad.material.uniforms.embeddings_u32.value = embeddingsDataTextureU32;

        //geometry.setFromPoints(results.map(x => new THREE.Vector3(x[0], x[1], 0)));
        geometry.setAttribute('position', new THREE.BufferAttribute(results, 3));

        let avgPosX = 0;
        let avgPosY = 0;
        for (let i = 0; i < results.length; i += 3) {
            avgPosX += results[i];
            avgPosY += results[i + 1];
        }
        avgPosX /= results.length / 3;
        avgPosY /= results.length / 3;
        let varPosX = 0;
        let varPosY = 0;
        for (let i = 0; i < results.length; i += 3) {
            varPosX += (results[i] - avgPosX) * (results[i] - avgPosX);
            varPosY += (results[i + 1] - avgPosY) * (results[i + 1] - avgPosY);
        }
        varPosX /= results.length / 3;
        varPosY /= results.length / 3;

        // camera.position.set(avgPosX, avgPosY, 5);
        // camera.lookAt(avgPosX, avgPosY, 0);
        // camera.zoom = 1;
        camera.position.x = avgPosX;
        camera.position.y = avgPosY;
        camera.position.z = 5;
        camera.zoom = 0.5 / Math.max(Math.sqrt(varPosX), Math.sqrt(varPosY));
        controls.target.set(avgPosX, avgPosY, 0);
        controls.update();
        timelineArr = new Float32Array(contexts.length);
        timelineArrUnnormalized = new Float32Array(contexts.length);
        geometry.setAttribute('intensity', new THREE.BufferAttribute(timelineArr, 1))

        const ids = new Float32Array(contexts.length);
        for (let i = 0; i < contexts.length; i++) {
            ids[i] = i;
        }
        geometry.setAttribute('id', new THREE.BufferAttribute(ids, 1));
        rendererGraph.render(scene, camera);
        doUpdate = true;
        requestAnimationFrame(animate);
    }
    setitup(embedding_u8, contextMinMax, contexts, sentenceRanges, results);

}
/*main({
    textFile: await (await fetch("frankenstein.txt")).text()
});*/

document.getElementById('fileInput').addEventListener('change', async(event) => {
    const txt = await readFile(event);
    main({ textFile: txt });
});

async function readFile(event) {
    const file = event.target.files[0];
    if (!file) return;

    if (file.type === 'application/pdf') {
        return await readPDF(file);
    } else if (file.type === 'text/plain') {
        return await readTextFile(file);
    }
}

function readTextFile(file) {
    const reader = new FileReader();
    /*reader.onload = function(e) {
        document.getElementById('content').textContent = e.target.result;
    };
    reader.readAsText(file);*/
    return new Promise((resolve, reject) => {
        reader.onload = function(e) {
            resolve(e.target.result);
        };
        reader.readAsText(file);
    });

}

function readPDF(file) {
    const fileReader = new FileReader();
    fileReader.readAsArrayBuffer(file);
    return new Promise((resolve, reject) => {
        fileReader.onload = function() {
            const typedarray = new Uint8Array(this.result);

            pdfjsLib.getDocument(typedarray).promise.then(pdf => {

                const numPages = pdf.numPages;
                const pagePromises = [];

                for (let pageNum = 1; pageNum <= numPages; pageNum++) {
                    pagePromises.push(getPageText(pageNum, pdf));
                }

                Promise.all(pagePromises).then(pagesText => {
                    pagesText = pagesText.map(s => s.replace(/\s+/g, ' '));
                    resolve(pagesText.join('\r\n'));
                });
            });
        };
    });
}

function getPageText(pageNum, PDFDocumentInstance) {
    // Return a Promise that is solved once the text of the page is retrievable
    return new Promise(function(resolve, reject) {
        PDFDocumentInstance.getPage(pageNum).then(function(pdfPage) {
            // The main trick to obtain the text of the PDF page, use the getTextContent method
            pdfPage.getTextContent().then(function(textContent) {
                const textItems = textContent.items;
                const finalString = textItems.map(item => item.str).join(" ");
                resolve(finalString);
            });
        });
    });
}

document.getElementById("loadFileButton").addEventListener('click', async() => {
    console.time();
    const loadfilePath = document.getElementById("loadfileTxt").value;
    const url = 'http://localhost:3000/load';

    const dataWoutEmbeddings = await (await (fetch(url, {
        method: 'POST', // or 'PUT'
        headers: {
            'Content-Type': 'application/json',
            noCors: true
        },

        body: JSON.stringify({ filename: loadfilePath }), // Convert JavaScript object to a JSON string
    }))).json();

    /* const embeddings = new Float32Array(await (await (fetch(urlEmbeddings, {
         method: 'POST', // or 'PUT'
         headers: {
             'Content-Type': 'application/json',
             noCors: true
         },

         body: JSON.stringify({ filename: loadfilePath }), // Convert JavaScript object to a JSON string
     }))).arrayBuffer());*/

    const embeddings_u8 = new Uint8Array(await (await (fetch('http://localhost:3000/load-embeddings-u8', {
        method: 'POST', // or 'PUT'
        headers: {
            'Content-Type': 'application/json',
            noCors: true
        },

        body: JSON.stringify({ filename: loadfilePath }), // Convert JavaScript object to a JSON string
    }))).arrayBuffer());

    const contextMinMax = new Float32Array(await (await (fetch('http://localhost:3000/load-embeddings-ctx', {
        method: 'POST', // or 'PUT'
        headers: {
            'Content-Type': 'application/json',
            noCors: true
        },

        body: JSON.stringify({ filename: loadfilePath }), // Convert JavaScript object to a JSON string
    }))).arrayBuffer());

    let results = new Float32Array(await (await (fetch('http://localhost:3000/load-results', {
        method: 'POST', // or 'PUT'
        headers: {
            'Content-Type': 'application/json',
            noCors: true
        },

        body: JSON.stringify({ filename: loadfilePath }), // Convert JavaScript object to a JSON string
    }))).arrayBuffer());
    /*let newResults = [];
    for (let i = 0; i < results.length; i += 3) {
        newResults.push([results[i], results[i + 1], results[i + 2]]);
    }*/


    const { sentences, WINDOW_SIZE, WINDOW_STRIDE } = dataWoutEmbeddings;
    const { contexts, sentenceRanges } = computeContextAndSentenceRanges(sentences, WINDOW_SIZE, WINDOW_STRIDE);
    const data = {
        //embeddings: embeddings,
        embedding_u8: embeddings_u8,
        contextMinMax: contextMinMax,
        contexts: contexts,
        sentenceRanges: sentenceRanges,
        results: results,
        sentences: sentences
    };
    main({ loadFile: true, data: data });
    console.timeEnd();

});