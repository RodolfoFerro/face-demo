let video = document.getElementById('video');
let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
let faceModel;
let tfliteModel;
let emoji = document.createElement('img');

const emotions = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Sad',
    'Surprise',
    'Neutral'
];

const emojis = [
    'assets/emojis/angry.png',
    'assets/emojis/disgust.png',
    'assets/emojis/fear.png',
    'assets/emojis/happy.png',
    'assets/emojis/sad.png',
    'assets/emojis/surprise.png',
    'assets/emojis/neutral.png'
]

function argMax(array) {
    return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}


const setupWebcam = () => {
    navigator.mediaDevices
        .getUserMedia({
            video: {
                width: 300,
                height: 300
            },
            audio: false 
        })
        .then(stream => {
            video.srcObject = stream;
        });
};


const displayEmojis = async () => {
    const prediction = await faceModel.estimateFaces(video, false);
    
    ctx.drawImage(video, 0, 0, 300, 300);
    prediction.forEach((pred) => {
        ctx.strokeStyle = '#03fc9d';
        ctx.lineWidth = 2;
        ctx.strokeRect(
            pred.topLeft[0],
            pred.topLeft[1] - 30,
            pred.bottomRight[0] - pred.topLeft[0],
            pred.bottomRight[1] - pred.topLeft[1] + 30
        );
        
        const img = tf.browser
            .fromPixels(video);

        const crop = img.slice(
            [
                Math.floor(pred.topLeft[1] - 30),
                Math.floor(pred.topLeft[0])
            ],
            [
                Math.floor(pred.bottomRight[1] - pred.topLeft[1] + 30),
                Math.floor(pred.bottomRight[0] - pred.topLeft[0])
            ])
            .expandDims()
            .resizeNearestNeighbor([48, 48])
            .mean(3)
					.expandDims(-1);
			        
			const outputTensor = tfliteModel.predict(crop);
			const emotion = Array.from((outputTensor.dataSync()));
			const prediction = argMax(emotion);
			
			emoji.src = emojis[prediction];
			const scale = Math.floor((pred.bottomRight[0] - pred.topLeft[0]) / 4);
        ctx.drawImage(
            emoji,
            Math.floor(pred.bottomRight[0] - scale - 3),
            Math.floor(pred.topLeft[1] - 27),
            scale, scale
        );
    });
};


setupWebcam();

video.addEventListener('loadeddata', async () => {
	faceModel = await blazeface.load();
	tfliteModel = await tflite.loadTFLiteModel('assets/tf_model.tflite');
	setInterval(displayEmojis, 100);
});