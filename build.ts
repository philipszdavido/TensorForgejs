import * as esbuild from 'esbuild';
import {execSync} from 'child_process';

async function build() {

    await esbuild.build({
        entryPoints: ['src/index.ts'],
        bundle: true,
        minify: false,
        format: 'esm',
        outfile: 'dist/TensorForge.js',
    });

    console.log('Generating type definitions...');
    execSync('npx dts-bundle-generator -o dist/TensorForge.d.ts src/index.ts');

    console.log('🚀 Build ready!');
}

build();
