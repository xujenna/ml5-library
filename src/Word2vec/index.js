// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/*
Word2Vec
*/
// import * as similarity from 'compute-cosine-similarity';
import * as tf from '@tensorflow/tfjs';
import callCallback from '../utils/callcallback';

// const similarity = require('compute-cosine-similarity');

class Word2Vec {
  constructor(modelPath, callback) {
    this.model = {};
    this.modelPath = modelPath;
    this.modelSize = 0;
    this.modelLoaded = false;

    this.ready = callCallback(this.loadModel(), callback);
    // TODO: Add support to Promise
    // this.then = this.ready.then.bind(this.ready);
  }

  async loadModel() {
    const json = await fetch(this.modelPath)
      .then(response => response.json());
    Object.keys(json.vectors).forEach((word) => {
      this.model[word] = tf.tensor1d(json.vectors[word]);
    });
    this.modelSize = Object.keys(this.model).length;
    this.modelLoaded = true;
    return this;
  }

  dispose(callback) {
    Object.values(this.model).forEach(x => x.dispose());
    if (callback) {
      callback();
    }
  }

  async add(inputs, maxOrCb, cb) {
    const { max, callback } = Word2Vec.parser(maxOrCb, cb, 10);

    await this.ready;
    return tf.tidy(() => {
      const sum = Word2Vec.operate(this.model, inputs, 'ADD');
      const nearest = Word2Vec.nearest(this.model, sum, inputs.length, inputs.length + max);
      const result = {
        vector_result: sum.dataSync(),
        nearest_words: nearest,
      };
      if (callback) {
        callback(undefined, result);
      }
      return result;
    });
  }

  async subtract(inputs, maxOrCb, cb) {
    const { max, callback } = Word2Vec.parser(maxOrCb, cb, 10);

    await this.ready;
    return tf.tidy(() => {
      const subtraction = Word2Vec.operate(this.model, inputs, 'SUBTRACT');
      const nearest = Word2Vec.nearest(this.model, subtraction, inputs.length, inputs.length + max);
      const result = {
        vector_result: subtraction.dataSync(),
        nearest_words: nearest,
      };
      if (callback) {
        callback(undefined, result);
      }
      return result;
    });
  }

  async divide(inputs, maxOrCb, cb) {
    const { max, callback } = Word2Vec.parser(maxOrCb, cb, 10);
    await this.ready;
    return tf.tidy(() => {
      const division = Word2Vec.operate(this.model, inputs, 'DIVIDE');
      const nearest = Word2Vec.nearest(this.model, division, inputs.length, inputs.length + max);
      const result = {
        vector_result: division.dataSync(),
        nearest_words: nearest,
      };
      if (callback) {
        callback(undefined, result);
      }
      return result;
    });
  }

  async multiply(inputs, maxOrCb, cb) {
    const { max, callback } = Word2Vec.parser(maxOrCb, cb, 10);
    await this.ready;
    return tf.tidy(() => {
      const product = Word2Vec.operate(this.model, inputs, 'MULTIPLY');
      const nearest = Word2Vec.nearest(this.model, product, inputs.length, inputs.length + max);
      const result = {
        vector_result: product.dataSync(),
        nearest_words: nearest,
      };
      if (callback) {
        callback(undefined, result);
      }
      return result;
    });
  }

  async average(inputs, maxOrCb, cb) {
    const { max, callback } = Word2Vec.parser(maxOrCb, cb, 10);

    await this.ready;
    return tf.tidy(() => {
      const sum = Word2Vec.operate(this.model, inputs, 'ADD');
      const avg = tf.div(sum, tf.tensor(inputs.length));
      const result = Word2Vec.nearest(this.model, avg, inputs.length, inputs.length + max);
      if (callback) {
        callback(undefined, result);
      }
      return result;
    });
  }

  async nearest(input, maxOrCb, cb) {
    const { max, callback } = Word2Vec.parser(maxOrCb, cb, 10);

    await this.ready;
    const vector = this.model[input];
    let result;
    if (vector) {
      result = Word2Vec.nearest(this.model, vector, 0, max);
    } else {
      result = null;
    }

    if (callback) {
      callback(undefined, result);
    }
    return result;
  }

  async getRandomWord(callback) {
    await this.ready;
    const words = Object.keys(this.model);
    const result = words[Math.floor(Math.random() * words.length)];
    if (callback) {
      callback(undefined, result);
    }
    return result;
  }

  static parser(maxOrCallback, cb, defaultMax) {
    let max = defaultMax;
    let callback = cb;

    if (typeof maxOrCallback === 'function') {
      callback = maxOrCallback;
    } else if (typeof maxOrCallback === 'number') {
      max = maxOrCallback;
    }
    return { max, callback };
  }

  static operate(model, values, operation) {
    return tf.tidy(() => {
      const vectors = [];
      const notFound = [];
      if (values.length < 2) {
        throw new Error('Invalid input, must be passed more than 1 value');
      }
      values.forEach((value) => {
        if (typeof value === 'string') {
          const vector = model[value];
          if (!vector) {
            notFound.push(value);
          } else {
            vectors.push(vector);
          }
        } else if (typeof value === 'number') {
          vectors.push(tf.scalar(value));
        } else if (typeof value === 'object') {
          vectors.push(value);
        }
      });

      if (notFound.length > 0) {
        throw new Error(`Invalid input, vector not found for: ${notFound.toString()}`);
      }
      let result = vectors[0];
      if (operation === 'ADD') {
        for (let i = 1; i < vectors.length; i += 1) {
          result = tf.add(result, vectors[i]);
        }
      } else if (operation === 'SUBTRACT') {
        for (let i = 1; i < vectors.length; i += 1) {
          result = tf.sub(result, vectors[i]);
        }
      } else if (operation === 'DIVIDE') {
        for (let i = 1; i < vectors.length; i += 1) {
          result = tf.div(result, vectors[i]);
        }
      } else if (operation === 'MULTIPLY') {
        for (let i = 1; i < vectors.length; i += 1) {
          result = tf.mul(result, vectors[i]);
        }
      }
      return result;
    });
  }

  static nearest(model, input, start, max) {
    const nearestVectors = [];
    Object.keys(model).forEach((word) => {
      // const distance = tf.losses.cosineDistance(input, model[word]);
      const vector = model[word].dataSync();
      // const distance = tf.util.distSquared(input.dataSync(), model[word].dataSync());
      // const cosineSimilarity = similarity(input.dataSync(), model[word].dataSync());
      const x = input.dataSync();
      const y = model[word].dataSync();

      const cosineSimilarity = tf.dot(x, y) / (tf.sqrt(tf.dot(x, x)) * tf.sqrt(tf.dot(y, y)));
      nearestVectors.push({ word, cosineSimilarity, vector });
    });
    nearestVectors.sort((a, b) => a.distance - b.distance);
    return nearestVectors.slice(start, max);
  }
}

const word2vec = (model, cb) => new Word2Vec(model, cb);

export default word2vec;
