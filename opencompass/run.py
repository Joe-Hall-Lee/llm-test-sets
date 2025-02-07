from opencompass.cli.main import main
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if __name__ == '__main__':
    main()
