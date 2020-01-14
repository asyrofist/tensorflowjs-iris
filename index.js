/* library & json data */
const tf = require('@tensorflow/tfjs')
const train = require('./iris.json')
const test = require('./iris-testing.json')

/* output data */
const outputData = tf.tensor2d(
  train.map(item => [
    item.species === 'setosa' ? 1 : 0,
    item.species === 'virginica' ? 1 : 0,
    item.species === 'versicolor' ? 1 : 0,
  ])
)
// outputData.print()

/* training data */
const trainingData = tf.tensor2d(
  train.map(item => [
    item.sepal_length,
    item.sepal_width,
    item.petal_length,
    item.petal_width,
  ])
)
// trainingData.print()

/* testing data */
const test_tensor = tf.tensor2d(
  test.map(item => [
    item.sepal_length,
    item.sepal_width,
    item.petal_length,
    item.petal_width,
  ]),
  [3, 4]
)
// test_tensor.print()

/* modelling data */
const model = tf.sequential()
model.add(
  tf.layers.dense({
    inputShape: [4],
    activation: 'sigmoid',
    units: 5,
  })
)
model.add(
  tf.layers.dense({
    inputShape: [5],
    activation: 'sigmoid',
    units: 3,
  })
)
model.add(
  tf.layers.dense({
    activation: 'sigmoid',
    units: 3,
  })
)
model.compile({
  loss: 'meanSquaredError',
  optimizer: tf.train.adam(0.06),
})

/* prediction data */
model.fit(trainingData, outputData, { epochs: 100 })
const mi_test = tf.tensor2d([4.1, 2.8, 4.4, 1.5], [1, 4])
// mi_test.print()

// /* prediction data & log history losses */
// model.fit(trainingData, outputData, { epochs: 100 }).then(history => {
//   console.log(history)
// })
// const mi_test = tf.tensor2d([4.1, 2.8, 4.4, 1.5], [1, 4])
// mi_test.print()

/* output prediction */
const rr = model.predict(mi_test)
const nuevo = rr.dataSync()
const new_array = [nuevo[0], nuevo[1], nuevo[2]]
var indice = nuevo.indexOf(new_array.sort()[2])
if (indice == 0) {
  console.log('Iris setosa')
} else if (indice == 1) {
  console.log('Iris versicolor')
} else {
  console.log('Iris virginica')
}
