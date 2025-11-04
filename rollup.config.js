const resolve = require('@rollup/plugin-node-resolve').default;
const commonjs = require('@rollup/plugin-commonjs');
const typescript = require('@rollup/plugin-typescript');
const pkg = require('./package.json');

module.exports = {
  input: 'src/index.ts',
  external: [
    // keep dependencies external so consumers can install them
    ...Object.keys(pkg.dependencies || {}),
    // node builtins
    'fs', 'path', 'os', 'util', 'stream', 'events'
  ],
  plugins: [
    resolve({ preferBuiltins: true }),
    commonjs(),
    typescript({ tsconfig: './tsconfig.json' })
  ],
  output: {
    file: 'dist/index.js',
    format: 'cjs',
    sourcemap: true
  }
};
