let video = document.getElementById('video');
let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
let faceModel;
let tfliteModel;

const emotions = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Sad',
    'Surprise',
    'Neutral'
];

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


const detectFaces = async () => {
    const prediction = await faceModel.estimateFaces(video, false);
    
    ctx.drawImage(video, 0, 0, 300, 300);
    prediction.forEach((pred) => {
        // ctx.strokeStyle = '#00ff00';
        // ctx.lineWidth = 2;
        // ctx.strokeRect(
        //     pred.topLeft[0],
        //     pred.topLeft[1] - 50,
        //     pred.bottomRight[0] - pred.topLeft[0],
        //     pred.bottomRight[1] - pred.topLeft[1] + 50
        // );

        let img = tf.browser
            .fromPixels(video)
            // .div(255.)
            .expandDims()
            .resizeNearestNeighbor([48, 48])
            .mean(3);

        let crop = tf.tensor([[
            pred.topLeft[0],
            pred.topLeft[1] - 50,
            pred.bottomRight[0] - pred.topLeft[0],
            pred.bottomRight[1] - pred.topLeft[1] + 50
        ]]);

        // const inputTensor = tf.image
        //     .cropAndResize(
        //         img.expandDims(-1),
        //         boxes=crop,
        //         box_indices=[0],
        //         // cropSize=[48, 48],
        //         cropSize=[250, 250],
        //         method='bilinear'
        //     );

        // console.log(img);
        // console.log(Array.from(inputTensor.dataSync()));
        // tf.browser.toPixels(tf.squeeze(inputTensor), canvas);
        // tf.browser.toPixels(tf.squeeze(img), canvas);
        
        let outputTensor = tfliteModel.predict(img.expandDims(-1));
        let emotion = Array.from((outputTensor.dataSync()));
        let prediction = argMax(emotion);
        // const emotion = tf.argMax(outputTensor);
        // console.log(Array.from(emotion.dataSync()));
        console.log(emotions[prediction]);
        // console.log(emotion);
    });
};


setupWebcam();


video.addEventListener('loadeddata', async () => {
    faceModel = await blazeface.load();
    tfliteModel = await tflite.loadTFLiteModel('assets/tf_model.tflite');
    setInterval(detectFaces, 100);
    // detectFaces();
});