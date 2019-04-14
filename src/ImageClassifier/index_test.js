// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

const { imageClassifier } = ml5;

const DEFAULTS = {
  learningRate: 0.0001,
  hiddenUnits: 100,
  epochs: 20,
  numClasses: 2,
  batchSize: 0.4,
  topk: 3,
  alpha: 1,
  version: 1,
};

describe('imageClassifier', () => {
  let classifier;

  async function getImage() {
    const img = new Image();
    img.crossOrigin = true;
    img.src = 'https://ml5js.org/docs/assets/img/bird.jpg';
    await new Promise((resolve) => { img.onload = resolve; });
    return img;
  }

  async function getCanvas() {
    const img = await getImage();
    const canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    canvas.getContext('2d').drawImage(img, 0, 0);
    return canvas;
  }

  beforeEach(async () => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 5000;
    classifier = await imageClassifier('MobileNet', undefined, {});
  });

  it('Should create a classifier with all the defaults', async () => {
    expect(classifier.version).toBe(DEFAULTS.version);
    expect(classifier.alpha).toBe(DEFAULTS.alpha);
    expect(classifier.topk).toBe(DEFAULTS.topk);
    expect(classifier.ready).toBeTruthy();
  });

  describe('classify', () => {
    it('Should classify an image of a Robin', async () => {
      const img = await getImage();
      await classifier.classify(img)
        .then(results => expect(results[0].label).toBe('robin, American robin, Turdus migratorius'));
    });

    it('Should support p5 elements with an image on .elt', async () => {
      const img = await getImage();
      await classifier.classify({ elt: img })
        .then(results => expect(results[0].label).toBe('robin, American robin, Turdus migratorius'));
    });

    it('Should support HTMLCanvasElement', async () => {
      const canvas = await getCanvas();
      await classifier.classify(canvas)
        .then(results => expect(results[0].label).toBe('robin, American robin, Turdus migratorius'));
    });

    it('Should support p5 elements with canvas on .canvas', async () => {
      const canvas = await getCanvas();
      await classifier.classify({ canvas })
        .then(results => expect(results[0].label).toBe('robin, American robin, Turdus migratorius'));
    });

    it('Should support p5 elements with canvas on .elt', async () => {
      const canvas = await getCanvas();
      await classifier.classify({ elt: canvas })
        .then(results => expect(results[0].label).toBe('robin, American robin, Turdus migratorius'));
    });
  });
});
