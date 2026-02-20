# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability, please follow these steps:

1. **Do NOT** open a public issue
2. Email the maintainer directly with details
3. Allow time for the issue to be resolved before public disclosure

## Security Measures

This project implements the following security practices:

- No patient data is stored or logged
- All predictions are made in real-time
- No external API calls that could leak data
- Input validation on all user inputs
- No hardcoded credentials

## Data Privacy

- This tool does not collect or store any personal information
- All patient data processing happens locally
- No data is sent to external servers
- No analytics or tracking

## Best Practices

When deploying this application:

1. Use HTTPS in production
2. Set strong environment variables
3. Keep dependencies updated
4. Use a firewall to restrict access if needed
5. Regularly backup your data

## Disclaimer

This tool is for educational and research purposes only. It should not be the sole basis for clinical decisions. Always consult with qualified healthcare professionals.
