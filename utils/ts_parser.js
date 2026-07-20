const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// 1. Self-installer for TypeScript module dependency
try {
  require.resolve('typescript');
} catch (e) {
  try {
    console.warn("TypeScript not found locally. Installing dependency dynamically...");
    execSync('npm install typescript --save-dev', { cwd: path.dirname(__dirname), stdio: 'inherit' });
  } catch (err) {
    console.error("Failed to install TypeScript dependency:", err.message);
    process.exit(1);
  }
}

const ts = require('typescript');

// 2. Main parser logic
function parseTSFile(filePath) {
  if (!fs.existsSync(filePath)) {
    console.error(`File not found: ${filePath}`);
    process.exit(1);
  }

  const content = fs.readFileSync(filePath, 'utf-8');
  const sourceFile = ts.createSourceFile(filePath, content, ts.ScriptTarget.Latest, true);

  const metadata = {
    classes: [],
    functions: [],
    interfaces: [],
    hooks: [],
    imports: [],
    exports: [],
    symbol_references: [],
    complexity_estimate: 1
  };

  function walk(node) {
    // Count complexity (branching nodes)
    if (
      node.kind === ts.SyntaxKind.IfStatement ||
      node.kind === ts.SyntaxKind.ForStatement ||
      node.kind === ts.SyntaxKind.ForInStatement ||
      node.kind === ts.SyntaxKind.ForOfStatement ||
      node.kind === ts.SyntaxKind.WhileStatement ||
      node.kind === ts.SyntaxKind.DoStatement ||
      node.kind === ts.SyntaxKind.CatchClause ||
      node.kind === ts.SyntaxKind.ConditionalExpression || // ternary ? :
      node.kind === ts.SyntaxKind.BinaryExpression
    ) {
      if (node.kind === ts.SyntaxKind.BinaryExpression) {
        const op = node.operatorToken.kind;
        if (op === ts.SyntaxKind.AmpersandAmpersandToken || op === ts.SyntaxKind.BarBarToken || op === ts.SyntaxKind.QuestionQuestionToken) {
          metadata.complexity_estimate += 1;
        }
      } else {
        metadata.complexity_estimate += 1;
      }
    }

    // Extract Imports
    if (ts.isImportDeclaration(node)) {
      const moduleName = node.moduleSpecifier.text || node.moduleSpecifier.getText(sourceFile);
      metadata.imports.push(moduleName.replace(/['"]/g, ''));
    }

    // Extract Exports
    if (ts.isExportDeclaration(node) || ts.isExportAssignment(node)) {
      const exportName = node.getText(sourceFile);
      metadata.exports.push(exportName.trim());
    }

    // Extract Classes
    if (ts.isClassDeclaration(node)) {
      const className = node.name ? node.name.text : "AnonymousClass";
      const bases = [];
      if (node.heritageClauses) {
        for (const clause of node.heritageClauses) {
          for (const typeNode of clause.types) {
            bases.push(typeNode.getText(sourceFile));
          }
        }
      }

      const cls_info = {
        name: className,
        methods: [],
        bases: bases,
        docstring: ""
      };

      // Traverse class children for methods
      node.forEachChild(member => {
        if (ts.isMethodDeclaration(member)) {
          const mname = member.name.getText(sourceFile);
          const args = member.parameters.map(p => p.name.getText(sourceFile));
          const ret = member.type ? member.type.getText(sourceFile) : "void";
          cls_info.methods.push({
            name: mname,
            arguments: args,
            returns: ret
          });
        }
      });
      metadata.classes.push(cls_info);
    }

    // Extract Interfaces
    if (ts.isInterfaceDeclaration(node)) {
      const interfaceName = node.name.text;
      const members = node.members.map(m => m.name ? m.name.getText(sourceFile) : "");
      metadata.interfaces.push({
        name: interfaceName,
        members: members.filter(Boolean)
      });
    }

    // Extract Hook Calls
    if (ts.isCallExpression(node)) {
      const expressionText = node.expression.getText(sourceFile);
      if (/^use[A-Z]/.test(expressionText) || ['useState', 'useEffect', 'useContext', 'useRef', 'useMemo', 'useCallback', 'useReducer'].includes(expressionText)) {
        if (!metadata.hooks.includes(expressionText)) {
          metadata.hooks.push(expressionText);
        }
      }
    }

    // Extract Functional Components / Constants Arrow Functions
    if (ts.isVariableDeclaration(node) && node.initializer) {
      const isArrow = ts.isArrowFunction(node.initializer) || ts.isFunctionExpression(node.initializer);
      const name = node.name.getText(sourceFile);
      
      if (isArrow) {
        const isFC = /^[A-Z]/.test(name); // React component naming rule
        metadata.functions.push({
          name: name,
          arguments: node.initializer.parameters.map(p => p.name.getText(sourceFile)),
          returns: isFC ? "JSX.Element" : (node.type ? node.type.getText(sourceFile) : "unknown"),
          docstring: isFC ? "React Functional Component" : "Arrow Function"
        });
      }
    }

    // Extract Traditional Function Declarations
    if (ts.isFunctionDeclaration(node)) {
      const funcName = node.name ? node.name.text : "AnonymousFunction";
      const isFC = /^[A-Z]/.test(funcName);
      metadata.functions.push({
        name: funcName,
        arguments: node.parameters.map(p => p.name.getText(sourceFile)),
        returns: isFC ? "JSX.Element" : (node.type ? node.type.getText(sourceFile) : "unknown"),
        docstring: isFC ? "React Functional Component" : "Function Declaration"
      });
    }

    // Symbol reference tracking
    if (ts.isIdentifier(node)) {
      const name = node.text;
      // Exclude keywords and standard builtins
      if (!['if', 'for', 'while', 'const', 'let', 'var', 'function', 'class', 'import', 'export', 'return', 'true', 'false', 'null', 'undefined', 'console', 'window', 'document'].includes(name)) {
        if (!metadata.symbol_references.includes(name)) {
          metadata.symbol_references.push(name);
        }
      }
    }

    ts.forEachChild(node, walk);
  }

  walk(sourceFile);
  return metadata;
}

// 3. Execution
if (process.argv.length < 3) {
  console.error("Usage: node ts_parser.js <filepath>");
  process.exit(1);
}

const targetPath = process.argv[2];
const parsedJSON = parseTSFile(targetPath);
console.log(JSON.stringify(parsedJSON, null, 2));
